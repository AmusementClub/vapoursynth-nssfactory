// Included inside nss::HWY_NAMESPACE after highway.h. No include guard: foreach_target
// re-includes the TU once per ISA. Generated codelets receive the Highway tag as d.
#define R float
#define V hn::Vec<decltype(d)>
#define DVK(name, value) const auto name = hn::Set(d, static_cast<float>(value))
#define LDK(name) (name)
#define LD(ptr, vstride, alignment) hn::LoadU(d, (ptr))
#define ST(ptr, value, vstride, alignment) hn::StoreU((value), d, (ptr))
#define VADD(a, b) hn::Add((a), (b))
#define VSUB(a, b) hn::Sub((a), (b))
#define VMUL(a, b) hn::Mul((a), (b))
#define VFMA(a, b, c) hn::MulAdd((a), (b), (c))
#define VFMS(a, b, c) hn::MulSub((a), (b), (c))
#define VFNMS(a, b, c) hn::NegMulAdd((a), (b), (c))
#define VNEG(a) hn::Neg((a))
#define VLEAVE() ((void)0)
