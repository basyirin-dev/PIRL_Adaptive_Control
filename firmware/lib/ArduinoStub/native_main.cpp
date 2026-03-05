// Compiled only when NATIVE_MAIN is defined (i.e. `pio run -e native`).
// Unity test runner provides its own main() for `pio test -e native_test`.
#ifdef NATIVE_MAIN

void setup();
void loop();

int main() {
    setup();
    loop();
    return 0;
}

#endif // NATIVE_MAIN
