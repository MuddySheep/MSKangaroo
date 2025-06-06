#include "GpuKang.h"
int main() {
    static_assert(sizeof(TPointPriv) == 96, "TPointPriv size mismatch");
    return 0;
}
