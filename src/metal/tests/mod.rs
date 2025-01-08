use std::sync::Once;

pub mod test_bn254;

// Static initializer to ensure the logger is initialized only once
static INIT: Once = Once::new();

pub(crate) fn init_logger() {
    INIT.call_once(|| {
        env_logger::builder()
            .is_test(true) // Ensures logs go to stdout/stderr in a test-friendly way
            .init();
    });
}
