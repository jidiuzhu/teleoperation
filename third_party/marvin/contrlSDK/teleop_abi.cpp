// Local adapter addition: verify ctypes layout before reading any hardware.
#include "FxRtCSDef.h"
#include <cstddef>
extern "C" size_t TeleopABI(int n) {
    const size_t sizes[] = {sizeof(DCSS), sizeof(StateCtr), sizeof(RT_IN), sizeof(RT_OUT),
        offsetof(DCSS,m_State), offsetof(DCSS,m_In), offsetof(DCSS,m_Out),
        offsetof(DCSS,m_ParaValueI), offsetof(DCSS,m_ParaRetSerial), sizeof(long)};
    return n >= 0 && n < 10 ? sizes[n] : 0;
}
