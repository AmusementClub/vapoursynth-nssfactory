// Isolated opcode-count controls. Not an application throughput benchmark.
.arch armv9-a+sve2
.text
.global nss_neon_instruction_loop
.type nss_neon_instruction_loop,%function
nss_neon_instruction_loop:
    movi v0.4s,#0
    movi v1.4s,#0
    movi v2.4s,#0
    movi v3.4s,#0
    fmov v16.4s,#1.0
    movi v17.4s,#0
1:  fmla v0.4s,v16.4s,v17.4s
    fmla v1.4s,v16.4s,v17.4s
    fmla v2.4s,v16.4s,v17.4s
    fmla v3.4s,v16.4s,v17.4s
    subs x0,x0,#1
    b.ne 1b
    ret
.size nss_neon_instruction_loop,.-nss_neon_instruction_loop
.global nss_sve_instruction_loop
.type nss_sve_instruction_loop,%function
nss_sve_instruction_loop:
    ptrue p0.s
    mov z0.s,#0
    mov z1.s,#0
    mov z2.s,#0
    mov z3.s,#0
    fmov z16.s,#1.0
    mov z17.s,#0
2:  fmla z0.s,p0/m,z16.s,z17.s
    fmla z1.s,p0/m,z16.s,z17.s
    fmla z2.s,p0/m,z16.s,z17.s
    fmla z3.s,p0/m,z16.s,z17.s
    subs x0,x0,#1
    b.ne 2b
    ret
.size nss_sve_instruction_loop,.-nss_sve_instruction_loop
