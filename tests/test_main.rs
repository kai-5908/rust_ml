#[test]
fn test_main() {
    assert!(1 == 1);
    assert_ne!(1, 2);
}

#[test]
#[should_panic]
fn test_panic_works() {
    panic!();
}
