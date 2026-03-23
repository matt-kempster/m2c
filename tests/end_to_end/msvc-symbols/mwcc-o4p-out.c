extern s32 ?TheDebug@@3VDebug@@A;
static s32 normalSymbol;

s32 test(void) {
    return ?TheDebug@@3VDebug@@A + normalSymbol;
}
