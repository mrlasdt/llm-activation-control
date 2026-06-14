import sys, pathlib, numpy as np, torch
H=pathlib.Path('/home/will/work/llm-activation-control/research_proposals/MOSAIC')
for p in [H.parents[1]/'pytorch_pure', H.parents[0]/'CASA', H]:
    sys.path.insert(0,str(p))
from utils import get_input_data
from mosaic_phase0 import load_model, load_band_pscale, HFScorer, fk_grade, distinct2, push_hooks, SENT_MODEL, FORM_MODEL, FORMAL_EX, INFORMAL_EX, POS_EX, NEG_EX
dev='cuda'
model,tok=load_model('google/gemma-2-2b-it',dev); band,ps=load_band_pscale(model)
form=HFScorer(FORM_MODEL,FORMAL_EX,INFORMAL_EX,dev)
z=np.load(H/'outputs'/'mosaic_cones_gemma-2-2b-it.npz'); bandz=list(z['band'])
dimU=z['formality_dim_U']; plsU=z['formality_pls_U']; plsa=z['formality_pls_a']  # (L,1,d),(L,k,d),(L,k)
def dirs(kind):
    g={}
    for i,l in enumerate(bandz):
        if kind=='dim': v=dimU[i,0]
        else: v=plsa[i]@plsU[i]   # intensity gradient in activation space
        g[l]=v/(np.linalg.norm(v)+1e-9)
    return g
_,test=get_input_data('harmless'); prompts=test[:16]
def run(g,m):
    vec={l:torch.from_numpy(m*g[l]).float() for l in bandz}
    from utils import generate_completions
    r=[c['response'] for c in generate_completions(model,prompts,tok,fwd_hooks=push_hooks(model,bandz,vec,dev),batch_size=16,max_new_tokens=96,temperature=0.0)]
    return float(np.nanmean(form.score(r))), float(np.nanmean([fk_grade(t) for t in r])), float(np.nanmean([distinct2(t) for t in r]))
bF,bR,_=run({l:np.zeros(plsU.shape[-1]) for l in bandz},0.0)
print(f"baseline: formality={bF:.3f} reading(FK)={bR:.2f}")
print(f"{'dir':4} {'frac':5} {'ΔFormality':>11} {'ΔReading(FK)':>13} {'drift/gain':>11} {'dist2':>6}")
for kind in ('dim','pls'):
    g=dirs(kind)
    for frac in (0.05,0.1):
        F,R,d2=run(g,frac*ps)
        gain=F-bF; drift=R-bR
        ratio=drift/gain if abs(gain)>1e-3 else float('nan')
        print(f"{kind:4} {frac:<5} {gain:>+11.3f} {drift:>+13.2f} {ratio:>+11.1f} {d2:>6.2f}")
