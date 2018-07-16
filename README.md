# mips_to_c
Given a MIPS script, this program will convert it to pseudo-C. The goal is that eventually the output will be well-formed C, and eventually after that, byte-equivalent C.

Progress is still forthcoming. Here is a list of things that need to be done:

- [ ] Properly declare and name local variables
- [ ] Fix subroutine argument passing
- [ ] Improve output for struct member access
- [ ] Improve type-hint collection and output
- [ ] Support float literals
- [ ] Support double operations
- [ ] Support loops (currently the program will hang)
- [ ] Support ternary operators/weird casts that cause blocks to conditionally modify registers
- [ ] Improve if-statement handler to automatically output || and && when appropriate
- [ ] Improve handling of returns
- [ ] Improve automatic commenting
- [ ] Write tests and test harness
