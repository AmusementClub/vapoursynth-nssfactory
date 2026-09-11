// Compile/disassemble only. C4A execution requires HWCAP2_SME and is not assumed.
// This is an instruction canary, not a production ABI or streaming-mode wrapper.
.arch armv9.2-a+sme
.text
.global nss_sme_compile_canary
.type nss_sme_compile_canary, %function
nss_sme_compile_canary:
    smstart
    ptrue p0.s
    fmov z0.s, #1.0
    zero {za}
    fmopa za0.s, p0/m, p0/m, z0.s, z0.s
    smstop
    ret
.size nss_sme_compile_canary, .-nss_sme_compile_canary
