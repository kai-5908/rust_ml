#[test]
fn test_main() {
    assert_eq!(1, 1);
    assert_ne!(1, 2);
}

#[test]
#[should_panic]
fn test_panic_works() {
    panic!();
}
