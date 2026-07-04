double pow(double, double);
double fmod(double, double);

double test(double a, double b) {
    return pow(a, b) * 2.0 + fmod(a, b);
}
