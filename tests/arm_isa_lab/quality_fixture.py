"""Shared graph construction for frozen quality-capture and stage replay."""
import numpy as np
import vapoursynth as vs


def graph(core, data, case, *, pilot=False, reference=None):
    name=case['algorithm'];sigma=case['sigma'];color=case['color'];radius=case['radius']
    values=data[f'{case["fixture"]}_s{sigma:g}_{color}']
    length,planes,height,width=values.shape
    blank=core.std.BlankClip(width=width,height=height,length=length,format=vs.RGBS if color=='rgb' else vs.GRAYS)
    def fill(n,f):
        out=f.copy()
        for p in range(out.format.num_planes):np.asarray(out[p])[:]=values[n,p]
        return out
    source=core.std.ModifyFrame(blank,blank,fill)
    core.std.SetVideoCache(source,mode=1,fixedsize=1,maxsize=length)
    if name=='NLM':kw=dict(d=radius,a=2,s=4,h=max(.1,sigma/2.5),channels='Y')
    elif name=='LSSC':kw=dict(sigma=sigma,block_size=8,block_step=8)
    else:kw=dict(sigma=[sigma]*planes if name in ('MCWNNM','TWSC') else sigma,
                 block_size=8,block_step=8,group_size=16 if name=='NLH' else 8,bm_range=7,radius=radius)
    def filtered(ref=None):
        extra={} if ref is None else {'ref' if name=='BM3D' else 'rclip':ref}
        node=getattr(core.nss,name)(source,**kw,**extra)
        return core.nss.VAggregate(node,source,radius=radius) if radius and name not in ('NLM','LSSC') else node
    if reference is not None:
        def ref_fill(n,f):
            out=f.copy()
            for p in range(out.format.num_planes):np.asarray(out[p])[:]=reference[n,p]
            return out
        node=filtered(core.std.ModifyFrame(blank,blank,ref_fill))
    elif pilot:node=filtered()
    else:node=filtered(filtered()) if case['stage']=='two-stage' else filtered()
    return node
