use defmt::*;
use nalgebra::{Matrix, Matrix4, Const, Vector3, UnitQuaternion};

// Define custom types for fixed-size matrices and vectors

// nalgebra doesn't support all the sizes of vectors and matrices that we need out of the box
// but we can create custom sizes like this
type Vector7 = Matrix<f64, Const<7>, Const<1>, nalgebra::ArrayStorage<f64, 7, 1>>;
type Matrix7 = Matrix<f64, Const<7>, Const<7>, nalgebra::ArrayStorage<f64, 7, 7>>;
type Matrix3 = Matrix<f64, Const<3>, Const<3>, nalgebra::ArrayStorage<f64, 3, 3>>;
type Matrix3x7 = Matrix<f64, Const<3>, Const<7>, nalgebra::ArrayStorage<f64, 3, 7>>;
type Vector4 = Matrix<f64, Const<4>, Const<1>, nalgebra::ArrayStorage<f64, 4, 1>>;

pub const GRAVITY: f64 = 9.81; // Gravitational constant (m/s^2)

pub struct EKF {
    pub state: Vector7,                 // State vector: [q0, q1, q2, q3, bx, by, bz]
    pub covariance: Matrix7,            // Covariance matrix P
    pub process_noise: Matrix7,         // Process noise Q
    pub measurement_noise: Matrix3,     // Measurement noise R
}

impl EKF {
    /// Create a new EKF instance, passing accelerometer data to calculate the initial quaternion (avoids using 0's for initial orientation)
    pub fn new(accel_data: Option<[f64; 3]>) -> Self {
        let (q0, q1, q2, q3) = if let Some(accel_data) = accel_data {
            // Normalize accelerometer vector
            let norm = Vector3::new(accel_data[0], accel_data[1], accel_data[2]).norm();
            let ax = accel_data[0] / norm;
            let ay = accel_data[1] / norm;
            let az = -accel_data[2] / norm;
    
            // Calculate quaternion from accelerometer data
            let q0 = (1.0 + az).sqrt() / 2.0;
            let q1 = -ay / (2.0 * q0);
            let q2 = ax / (2.0 * q0);
            let q3: f64 = 0.0; // Yaw is zero since accelerometer data cannot calulcate yaw angles

            let norm = Vector4::new(q0, q1, q2, q3).norm();
            let q0 = q0 / norm;
            let q1 = q1 / norm;
            let q2 = q2 / norm;
            let q3 = q3 / norm;
            

            (q0, q1, q2, q3)
        } else {
            // Default to identity quaternion
            (1.0, 0.0, 0.0, 0.0)
        };
    
        // Initialize process and measurement noise matrices
        let mut process_noise = Matrix7::zeros();
        process_noise[(0, 0)] = 0.05; // q0
        process_noise[(1, 1)] = 0.05; // q1
        process_noise[(2, 2)] = 0.05; // q2
        process_noise[(3, 3)] = 0.05; // q3
        process_noise[(4, 4)] = 0.01; // bx
        process_noise[(5, 5)] = 0.01; // by
        process_noise[(6, 6)] = 0.01; // bz
        
        let mut measurement_noise = Matrix3::zeros();
        measurement_noise[(0, 0)] = 0.02; // accel x
        measurement_noise[(1, 1)] = 0.02; // accel y
        measurement_noise[(2, 2)] = 0.02; // accel z 

    
        EKF {
            state: {
                let mut state = Vector7::zeros();
                state[0] = q0;
                state[1] = q1;
                state[2] = q2;
                state[3] = q3;
                state[4] = 0.0; // bx
                state[5] = 0.0; // by
                state[6] = 0.0; // bz
                state
            },
            covariance: Matrix7::identity() * 1.0, // Initial state covariance
            process_noise,
            measurement_noise,
        }
    }
    
    /// EKF Predict Step: Propagates state and covariance forward using gyro data.
    pub fn predict(&mut self, gyro: [f64; 3], dt: f64) {
        // 1. Subtract estimated bias from raw gyro measurements (control input)
        let bias = self.state.fixed_rows::<3>(4).clone_owned(); // [bx, by, bz]
        let omega = Vector3::new(gyro[0], gyro[1], gyro[2]) - bias;

        // 2. Integrate quaternion using angular velocity (q̇ = 0.5 * Ω(ω) * q)
        let q = Vector4::new(self.state[0], self.state[1], self.state[2], self.state[3]);
        let omega_matrix = Self::omega_matrix(omega);
        let q_dot = 0.5 * omega_matrix * q;
        let q_new = q + q_dot * dt;

        // 3. Update state with new quaternion
        self.state.fixed_rows_mut::<4>(0).copy_from(&q_new);
        Self::normalize_quaternion_in_state(&mut self.state);

        // 4. Compute Jacobian of motion model (F = ∂f/∂x)
        let f_jacobian = self.compute_f_jacobian(gyro, dt);
        
        // 5. Propagate uncertainty using the covariance update: P' = FPFᵀ + Q
        self.covariance = f_jacobian * self.covariance * f_jacobian.transpose() + self.process_noise;

        // 6. Optional: lock yaw axis to ensure stability with only accelerometer
        self.lock_yaw(); // ✅ centralized yaw suppression
    }


    /// EKF Update Step: Corrects the prediction using accelerometer data (gravity vector).
    pub fn update(&mut self, accel: [f64; 3]) {
        // 1. Compute expected gravity vector in sensor frame (using estimated orientation)
        let gravity = Vector3::new(0.0, 0.0, -GRAVITY);
        let q = Vector4::new(self.state[0], self.state[1], self.state[2], self.state[3]);
        let r_transpose = Self::quaternion_to_rotation_matrix(q).transpose();
        let accel_expected = r_transpose * gravity;

        // 2. Compute innovation (measurement residual): z - h(x)
        let z = Vector3::new(accel[0], accel[1], accel[2]);
        let innovation = z - accel_expected;

        // 3. Compute Jacobian of measurement model: H = ∂h/∂x
        let h_jacobian = self.compute_h_jacobian(q);

        // 4. Compute innovation covariance: S = HPHᵀ + R
        let s = h_jacobian * self.covariance * h_jacobian.transpose() + self.measurement_noise;

        // 5. Compute Kalman gain: K = PHᵀS⁻¹
        if let Some(s_inv) = s.try_inverse() {
            let k = self.covariance * h_jacobian.transpose() * s_inv;

            // 6. Update state: x = x + K(z - h(x))
            self.state += k * innovation;

            // 7. Update covariance: P = (I - KH)P
            self.covariance = (Matrix7::identity() - k * h_jacobian) * self.covariance;

            // 8. Normalize quaternion again after update
            Self::normalize_quaternion_in_state(&mut self.state);

            // 9. Re-lock yaw axis if necessary
            self.lock_yaw(); // ✅ centralized yaw suppression
        } else {
            error!("Warning: Skipping EKF update — non-invertible innovation covariance.");
        }
    }

    



    /// Compute the dynamic Jacobian (∂f/∂x)
    fn compute_f_jacobian(&self, gyro: [f64; 3], dt: f64) -> Matrix7 {
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

        // Quaternion dynamics wrt quaternion
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

        // Quaternion dynamics wrt bias
        f[(0, 4)] =  0.5 * q1 * dt;
        f[(0, 5)] =  0.5 * q2 * dt;
        f[(0, 6)] =  0.5 * q3 * dt;

        f[(1, 4)] = -0.5 * q0 * dt;
        f[(1, 5)] = -0.5 * q3 * dt;
        f[(1, 6)] =  0.5 * q2 * dt;

        f[(2, 4)] =  0.5 * q3 * dt;
        f[(2, 5)] = -0.5 * q0 * dt;
        f[(2, 6)] = -0.5 * q1 * dt;

        f[(3, 4)] = -0.5 * q2 * dt;
        f[(3, 5)] =  0.5 * q1 * dt;
        f[(3, 6)] = -0.5 * q0 * dt;

        f
    }

    /// Compute the measurement Jacobian (∂h/∂x)
    fn compute_h_jacobian(&self, q: Vector4) -> Matrix3x7 {
        let q0 = q[0];
        let q1 = q[1];
        let q2 = q[2];
        let q3 = q[3];

        let mut h = Matrix3x7::zeros();
        h[(0, 0)] = 2.0 * (-GRAVITY * q2);
        h[(0, 1)] = 2.0 * (GRAVITY * q3);
        h[(0, 2)] = 2.0 * (-GRAVITY * q0);
        h[(0, 3)] = 2.0 * (GRAVITY * q1);

        h[(1, 0)] = 2.0 * (GRAVITY * q1);
        h[(1, 1)] = 2.0 * (GRAVITY * q0);
        h[(1, 2)] = 2.0 * (GRAVITY * q3);
        h[(1, 3)] = 2.0 * (GRAVITY * q2);

        h[(2, 0)] = 2.0 * (GRAVITY * q0);
        h[(2, 1)] = 2.0 * (-GRAVITY * q1);
        h[(2, 2)] = 2.0 * (-GRAVITY * q2);
        h[(2, 3)] = 2.0 * (-GRAVITY * q3);

        h
    }

    fn normalize_quaternion_in_state(state: &mut Vector7) {
        let q = Vector4::new(state[0], state[1], state[2], state[3]);
        let norm = q.norm();
        if norm > 0.0 {
            let q_normalized = q / norm;
            state.fixed_rows_mut::<4>(0).copy_from(&q_normalized);
        }
    }
    


    /// Convert quaternion to rotation matrix
    fn quaternion_to_rotation_matrix(q: Vector4) -> Matrix3 {
        let q0 = q[0];
        let q1 = q[1];
        let q2 = q[2];
        let q3 = q[3];

        Matrix3::new(
            1.0 - 2.0 * (q2 * q2 + q3 * q3),
            2.0 * (q1 * q2 - q0 * q3),
            2.0 * (q1 * q3 + q0 * q2),

            2.0 * (q1 * q2 + q0 * q3),
            1.0 - 2.0 * (q1 * q1 + q3 * q3),
            2.0 * (q2 * q3 - q0 * q1),

            2.0 * (q1 * q3 - q0 * q2),
            2.0 * (q2 * q3 + q0 * q1),
            1.0 - 2.0 * (q1 * q1 + q2 * q2),
        )
    }

    /// Compute omega matrix for quaternion dynamics
    fn omega_matrix(omega: Vector3<f64>) -> Matrix<f64, Const<4>, Const<4>, nalgebra::ArrayStorage<f64, 4, 4>> {
        Matrix4::new(
            0.0, -omega[0], -omega[1], -omega[2],
            omega[0], 0.0, omega[2], -omega[1],
            omega[1], -omega[2], 0.0, omega[0],
            omega[2], omega[1], -omega[0], 0.0,
        )
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
        self.remove_yaw_from_quaternion();
    }
    



    /// Get the fully updated state vector
    pub fn get_state(&self) -> Vector7 {
        self.state.clone() // Return a copy of the state vector
    }

}