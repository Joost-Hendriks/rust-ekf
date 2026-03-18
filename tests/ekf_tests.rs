use rust_ekf::{EKF, GRAVITY}; // Import EKF struct and GRAVITY constant

#[cfg(test)]
mod tests {
    use super::*;
    // We'll use assert_relative_eq from the approx crate for floating point comparisons.
    // Alternatively, you can define your own epsilon checks.
    use approx::assert_relative_eq;

    #[test]
    fn test_initialization() {
        // Test that the EKF initializes correctly with no accelerometer input (defaults to identity quaternion)
        let ekf = EKF::new(None, None);
        // The state vector should have 7 elements
        assert_eq!(ekf.state.len(), 7);
        // The quaternion should be initialized as identity: [1, 0, 0, 0]
        assert_relative_eq!(ekf.state[0], 1.0, epsilon = 1e-6);
        assert_relative_eq!(ekf.state[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(ekf.state[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(ekf.state[3], 0.0, epsilon = 1e-6);
        // Bias terms should be zero
        assert_relative_eq!(ekf.state[4], 0.0, epsilon = 1e-6);
        assert_relative_eq!(ekf.state[5], 0.0, epsilon = 1e-6);
        assert_relative_eq!(ekf.state[6], 0.0, epsilon = 1e-6);
        // Covariance and noise matrices should be square
        assert!(ekf.covariance.is_square());
        assert!(ekf.process_noise.is_square());
        assert!(ekf.measurement_noise.is_square());
    }

    #[test]
    fn test_predict() {
        // Test the predict step with sample gyro data and a fixed time step
        let mut ekf = EKF::new(None, None);
        let dt = 0.01;
        let gyro_data = [0.1, 0.2, 0.3]; // in rad/s
        let initial_state = ekf.get_state();

        ekf.predict(gyro_data, dt);
        let new_state = ekf.get_state();

        // Check that the quaternion has changed from the identity
        let quat_changed = (initial_state[0] - new_state[0]).abs() > 1e-6 ||
                           (initial_state[1] - new_state[1]).abs() > 1e-6 ||
                           (initial_state[2] - new_state[2]).abs() > 1e-6 ||
                           (initial_state[3] - new_state[3]).abs() > 1e-6;
        assert!(quat_changed, "Quaternion did not change after predict.");

        // Since we lock yaw, we expect the z bias (state[6]) to be 0
        assert_relative_eq!(new_state[6], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_update() {
        // Test that the update step correctly corrects the state using accelerometer data.
        let mut ekf = EKF::new(None, None);
        let dt = 0.01;
        let gyro_data = [0.1, 0.2, 0.3];
        // Run a predict to have a non-identity quaternion
        ekf.predict(gyro_data, dt);

        // Simulate an accelerometer measurement equivalent to gravity pointing along negative Z
        let accel_data = [0.0, 0.0, GRAVITY];
        ekf.update(accel_data);

        let new_state = ekf.get_state();
        // Verify that the quaternion remains normalized
        let norm = (new_state[0].powi(2)
            + new_state[1].powi(2)
            + new_state[2].powi(2)
            + new_state[3].powi(2))
            .sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }

}
