// Local adapter addition: verify ctypes IK structure layout before native calls.
#include "FXKineCommon.h"
#include <cstddef>
extern "C" size_t TeleopABI(int n) {
    const size_t sizes[] = {sizeof(FX_InvKineSolvePara),
        offsetof(FX_InvKineSolvePara,m_Input_IK_RefJoint),
        offsetof(FX_InvKineSolvePara,m_Output_RetJoint),
        offsetof(FX_InvKineSolvePara,m_Output_IsOutRange),
        offsetof(FX_InvKineSolvePara,m_Output_RunLmtP)};
    return n >= 0 && n < 5 ? sizes[n] : 0;
}
