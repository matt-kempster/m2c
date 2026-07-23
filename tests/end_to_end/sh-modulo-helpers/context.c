signed test(signed lhs, signed rhs);
unsigned test_unsigned(unsigned lhs, unsigned rhs);
signed __sdivsi3(signed lhs, signed rhs);
unsigned __udivsi3(unsigned lhs, unsigned rhs);

/* Real-world modulo shapes from a corpus; see manual.s. */
signed mod_extu_delay(signed lhs, signed rhs);
signed mod_split_dividend(signed lhs, signed rhs);
signed mod_feeds_multiply(signed lhs, signed rhs);
unsigned mod_moved_quotient(unsigned lhs, unsigned rhs);
unsigned mod_interleaved(unsigned lhs, unsigned rhs);
