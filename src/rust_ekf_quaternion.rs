use defmt::*;
use core::option::Option::{self, Some};
#[allow(unused_imports)]
use nalgebra::{Matrix, Const, Vector3, ComplexField, UnitQuaternion};

type Vector4 = Matrix<f32, Const<4>, Const<1>, nalgebra::ArrayStorage<f32, 4, 1>>;
type Vector7 = Matrix<f32, Const<7>, Const<1>, nalgebra::ArrayStorage<f32, 7, 1>>;
type Matrix3   = Matrix<f32, Const<3>, Const<3>, nalgebra::ArrayStorage<f32, 3, 3>>;
type Matrix3x4 = Matrix<f32, Const<3>, Const<4>, nalgebra::ArrayStorage<f32, 3, 4>>;
type Matrix7   = Matrix<f32, Const<7>, Const<7>, nalgebra::ArrayStorage<f32, 7, 7>>;

pub const GRAVITY: f32 = 9.81;

pub struct EKF {
    pub state: Vector7,                 // State vector: [q0, q1, q2, q3, bx, by, bz]
    pub covariance: Matrix7,            // Covariance matrix P
    pub process_noise: Matrix7,         // Process noise Q
    pub measurement_noise: Matrix3,     // Measurement noise R
}

impl EKF {
    /// Create a new EKF instance.
    /// `accel_data`: used to compute the initial quaternion from gravity direction.
    /// `gyro_bias`: initial gyro bias estimate from startup calibration.
    pub fn new(accel_data: Option<[f32; 3]>, gyro_bias: Option<[f32; 3]>) -> Self {
        let (q0, q1, q2, q3) = if let Some(accel_data) = accel_data {
            let norm = Vector3::new(accel_data[0], accel_data[1], accel_data[2]).norm();
            let ax = accel_data[0] / norm;
            let ay = accel_data[1] / norm;
            let az = -accel_data[2] / norm;

            let q0 = (1.0 + az).sqrt() / 2.0;
            let q1 = -ay / (2.0 * q0);
            let q2 = ax / (2.0 * q0);
            let q3: f32 = 0.0;

            let norm = Vector4::new(q0, q1, q2, q3).norm();
            (q0 / norm, q1 / norm, q2 / norm, q3 / norm)
        } else {
            (1.0, 0.0, 0.0, 0.0)
        };

        let mut process_noise = Matrix7::zeros();
        process_noise[(0, 0)] = 0.05;
        process_noise[(1, 1)] = 0.05;
        process_noise[(2, 2)] = 0.05;
        process_noise[(3, 3)] = 0.05;
        process_noise[(4, 4)] = 0.001;
        process_noise[(5, 5)] = 0.001;
        process_noise[(6, 6)] = 0.001;

        let mut measurement_noise = Matrix3::zeros();
        measurement_noise[(0, 0)] = 0.02;
        measurement_noise[(1, 1)] = 0.02;
        measurement_noise[(2, 2)] = 0.02;

        EKF {
            state: {
                let mut state = Vector7::zeros();
                state[0] = q0;
                state[1] = q1;
                state[2] = q2;
                state[3] = q3;
                if let Some(bias) = gyro_bias {
                    state[4] = bias[0];
                    state[5] = bias[1];
                    state[6] = bias[2];
                }
                state
            },
            covariance: Matrix7::identity(),
            process_noise,
            measurement_noise,
        }
    }

    /// EKF Predict Step: Propagates state and covariance forward using gyro data.
    pub fn predict(&mut self, gyro: [f32; 3], dt: f32) {
        // 1. Subtract estimated bias
        let bx = self.state[4];
        let by = self.state[5];
        let bz = self.state[6];
        let wx = gyro[0] - bx;
        let wy = gyro[1] - by;
        let wz = gyro[2] - bz;

        // 2. Integrate quaternion directly — avoids building the 4x4 Omega matrix
        //    q̇ = 0.5 * Ω(ω) * q  →  inlined as scalar ops (12 MACs vs 64)
        let (q0, q1, q2, q3) = (self.state[0], self.state[1], self.state[2], self.state[3]);
        let half_dt = 0.5 * dt;
        let q_new = Vector4::new(
            q0 + half_dt * (-wx * q1 - wy * q2 - wz * q3),
            q1 + half_dt * ( wx * q0 + wz * q2 - wy * q3),
            q2 + half_dt * ( wy * q0 - wz * q1 + wx * q3),
            q3 + half_dt * ( wz * q0 + wy * q1 - wx * q2),
        );
        self.state.fixed_rows_mut::<4>(0).copy_from(&q_new);
        Self::normalize_quaternion_in_state(&mut self.state);

        // 3. Compute F Jacobian (sparse: only top 4 rows differ from identity)
        let f = self.compute_f_jacobian(gyro, dt);

        // 4. Covariance update exploiting F = [A B; 0 I] block structure
        //    P' = F*P*F^T + Q  decomposes as:
        //      top-left  (4x4): (A*P11 + B*P21)*A^T + (A*P12 + B*P22)*B^T
        //      top-right (4x3): A*P12 + B*P22       (= temp2)
        //      bot-left  (3x4): temp2^T              (P is symmetric)
        //      bot-right (3x3): P22                  (identity rows → unchanged by F)
        //    ~308 MACs vs 686 for two full 7x7 multiplies
        let (new_p11, temp2, temp2_t, p22_owned) = {
            let a   = f.fixed_view::<4, 4>(0, 0);
            let b   = f.fixed_view::<4, 3>(0, 4);
            let p11 = self.covariance.fixed_view::<4, 4>(0, 0);
            let p12 = self.covariance.fixed_view::<4, 3>(0, 4);
            let p21 = self.covariance.fixed_view::<3, 4>(4, 0);
            let p22 = self.covariance.fixed_view::<3, 3>(4, 4);

            let temp1 = a * p11 + b * p21;  // 4x4
            let temp2 = a * p12 + b * p22;  // 4x3

            let new_p11 = temp1 * a.transpose() + &temp2 * b.transpose();
            let temp2_t = temp2.transpose();
            let p22_owned = p22.clone_owned();
            (new_p11, temp2, temp2_t, p22_owned)
        };

        // Assemble p_new = F*P*F^T + Q, starting from Q (its off-diagonal blocks are zero)
        let mut p_new = self.process_noise;
        let q11 = p_new.fixed_view::<4, 4>(0, 0).clone_owned();
        let q22 = p_new.fixed_view::<3, 3>(4, 4).clone_owned();
        p_new.fixed_view_mut::<4, 4>(0, 0).copy_from(&(new_p11 + q11));
        p_new.fixed_view_mut::<4, 3>(0, 4).copy_from(&temp2);
        p_new.fixed_view_mut::<3, 4>(4, 0).copy_from(&temp2_t);
        p_new.fixed_view_mut::<3, 3>(4, 4).copy_from(&(p22_owned + q22));
        self.covariance = p_new;

        // self.lock_yaw();
    }

    /// EKF Update Step: Corrects the prediction using accelerometer data (gravity vector).
    pub fn update(&mut self, accel: [f32; 3]) {
        let mag = (accel[0]*accel[0] + accel[1]*accel[1] + accel[2]*accel[2]).sqrt();
        let deviation = (mag - GRAVITY).abs();
        
        // Scale R: at rest deviation≈0 → scale=1; under 2g vibration → scale~25x
        let r_scale = 1.0 + (deviation / 0.5).powi(2);
        let r_scaled = self.measurement_noise * r_scale;

        // 1. Compute expected gravity directly from quaternion — avoids building R and transposing
        //    h(x) = R^T * [0, 0, -g], expanded inline from the third row of R
        let (q0, q1, q2, q3) = (self.state[0], self.state[1], self.state[2], self.state[3]);
        let accel_expected = Vector3::new(
            2.0 * GRAVITY * (q0 * q2 - q1 * q3),
            -2.0 * GRAVITY * (q2 * q3 + q0 * q1),
            GRAVITY * (2.0 * (q1 * q1 + q2 * q2) - 1.0),
        );

        // 2. Innovation
        let z = Vector3::new(accel[0], accel[1], accel[2]);
        let innovation = z - accel_expected;

        // 3. H is 3x4 — bias columns are always zero, excluded to reduce work
        let h = self.compute_h_jacobian(Vector4::new(q0, q1, q2, q3));

        // 4. S = H * P11 * H^T + R  — only the quaternion block of P matters (~84 MACs vs 210)
        let p11 = self.covariance.fixed_view::<4, 4>(0, 0).clone_owned();
        let s = h * p11 * h.transpose() + r_scaled;

        if let Some(s_inv) = s.try_inverse() {
            // 5. K = P[:, 0:4] * H^T * S^-1  — only left 4 columns of P needed (~147 MACs vs 210)
            let p_left = self.covariance.fixed_view::<7, 4>(0, 0).clone_owned();
            let k = p_left * h.transpose() * s_inv;  // 7x3

            // 6. State update
            self.state += k * innovation;

            // 7. Covariance update: P = P - (K*H)*P[0:4,:]
            //    Equivalent to (I - K*H_full)*P but avoids the full 7x7 K*H product
            //    ~196 MACs vs ~490 for the naive form
            let kh = k * h;  // 7x4
            let p_top = self.covariance.fixed_view::<4, 7>(0, 0).clone_owned();
            self.covariance -= kh * p_top;

            Self::normalize_quaternion_in_state(&mut self.state);
            // self.lock_yaw();
        } else {
            error!("Warning: Skipping EKF update — non-invertible innovation covariance.");
        }
    }

    /// Compute the dynamic Jacobian (∂f/∂x)
    fn compute_f_jacobian(&self, gyro: [f32; 3], dt: f32) -> Matrix7 {
        let q0 = self.state[0];
        let q1 = self.state[1];
        let q2 = self.state[2];
        let q3 = self.state[3];
        let bx = self.state[4];
        let by = self.state[5];
        let bz = self.state[6];

        let p = gyro[0] - bx;
        let q = gyro[1] - by;
        let r = gyro[2] - bz;

        let mut f = Matrix7::identity();

        f[(0, 1)] = -p * dt;
        f[(0, 2)] = -q * dt;
        f[(0, 3)] = -r * dt;

        f[(1, 0)] =  p * dt;
        f[(1, 2)] =  r * dt;
        f[(1, 3)] = -q * dt;

        f[(2, 0)] =  q * dt;
        f[(2, 1)] = -r * dt;
        f[(2, 3)] =  p * dt;

        f[(3, 0)] =  r * dt;
        f[(3, 1)] =  q * dt;
        f[(3, 2)] = -p * dt;

        f[(0, 4)] =  0.5 * q1 * dt;
        f[(0, 5)] =  0.5 * q2 * dt;
        f[(0, 6)] =  0.5 * q3 * dt;

        f[(1, 4)] = -0.5 * q0 * dt;
        f[(1, 5)] =  0.5 * q3 * dt;
        f[(1, 6)] = -0.5 * q2 * dt;

        f[(2, 4)] = -0.5 * q3 * dt;
        f[(2, 5)] = -0.5 * q0 * dt;
        f[(2, 6)] =  0.5 * q1 * dt;

        f[(3, 4)] =  0.5 * q2 * dt;
        f[(3, 5)] = -0.5 * q1 * dt;
        f[(3, 6)] = -0.5 * q0 * dt;

        f
    }

    /// Compute the measurement Jacobian (∂h/∂x) — returns only the 3x4 quaternion columns.
    /// The 3x3 bias columns are always zero and are excluded.
    fn compute_h_jacobian(&self, q: Vector4) -> Matrix3x4 {
        let (q0, q1, q2, q3) = (q[0], q[1], q[2], q[3]);

        let mut h = Matrix3x4::zeros();
        h[(0, 0)] =  2.0 * GRAVITY * q2;
        h[(0, 1)] = -2.0 * GRAVITY * q3;
        h[(0, 2)] =  2.0 * GRAVITY * q0;
        h[(0, 3)] = -2.0 * GRAVITY * q1;

        h[(1, 0)] = -2.0 * GRAVITY * q1;
        h[(1, 1)] = -2.0 * GRAVITY * q0;
        h[(1, 2)] = -2.0 * GRAVITY * q3;
        h[(1, 3)] = -2.0 * GRAVITY * q2;

        h[(2, 1)] = 4.0 * GRAVITY * q1;
        h[(2, 2)] = 4.0 * GRAVITY * q2;

        h
    }

    fn normalize_quaternion_in_state(state: &mut Vector7) {
        let q = Vector4::new(state[0], state[1], state[2], state[3]);
        let norm = q.norm();
        if norm > 0.0 {
            state.fixed_rows_mut::<4>(0).copy_from(&(q / norm));
        }
    }

    fn remove_yaw_from_quaternion(&mut self) {
        let q = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            self.state[0], self.state[1], self.state[2], self.state[3],
        ));
    
        let euler = q.euler_angles();
        let new_q = UnitQuaternion::from_euler_angles(euler.0, euler.1, 0.0);
        let qn = new_q.quaternion();
    
        self.state[0] = qn.w;
        self.state[1] = qn.i;
        self.state[2] = qn.j;
        self.state[3] = qn.k;
    }

    fn lock_yaw(&mut self) {
        self.state[6] = 0.0;
        self.covariance[(6, 6)] = 0.0;
        // self.remove_yaw_from_quaternion();
    }


    /// Get the fully updated state vector
    pub fn get_state(&self) -> Vector7 {
        self.state // Return a copy of the state vector
    }

}
