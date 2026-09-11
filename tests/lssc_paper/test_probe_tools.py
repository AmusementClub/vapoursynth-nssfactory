import numpy as np

from author_probe import driver, quote
from benchmark import cpu_delta, eligible, stage_costs
from reference import denoise
from author_budgets import make_driver, parse_cpu, schedule


def test_octave_quote_and_driver_do_not_pass_ground_truth():
    assert quote("a'b")=="'a''b'"
    script=driver('/external','/temporary','/output',sigma=.1,block=9,j1=0,j2=0,
                  window=1,threshold=0,warmups=1,repeats=3)
    assert 'mexDenoise(I,I,D,' in script
    assert 'for pass=1:4' in script
    assert 'elapsed=toc;' in script
    assert 'Dout(:)-D(:)' in script
    assert 'num2hex(I(:))' in script and 'num2hex(D(:))' in script


def test_cpu_accounting_keeps_missing_distinct_from_idle():
    assert cpu_delta(None,None) is None
    assert not eligible(None)
    before={'cpu0':[0]*8,'cpu1':[0]*8}
    after={'cpu0':[100,0,0,0,0,0,0,0],'cpu1':[0,0,0,100,0,0,0,0]}
    result=cpu_delta(before,after)
    assert eligible(result)
    after['cpu1']=[10,0,0,90,0,0,0,0]
    assert not eligible(cpu_delta(before,after))


def test_stage_instrumentation_preserves_output_and_restores_functions():
    import reference
    from benchmark import digest
    saved=reference.encode_groups
    image=.4+np.random.default_rng(4).normal(0,.03,(7,8))
    expected=denoise(image,np.eye(9),.03)
    measured=stage_costs(image,np.eye(9),.03)
    assert measured['output_sha256']==digest(expected.output)
    assert reference.encode_groups is saved
    assert 'encode_groups_0' in measured['components']
    assert 'encode_groups_1' in measured['components']


def test_author_budget_ladder_is_serial_alternating_and_same_window():
    assert schedule(2, 2) == [(-1, 0), (-1, 1), (0, 0), (0, 1), (1, 1), (1, 0)]
    script = make_driver('/external', '/tmp/probe', '/out', sigma=.1, block=9, pairs=2,
                         policies=[('fixed', 0, 0), ('full', 20, 5)])
    assert 'mexDenoise(I,I,D,' in script
    assert ',32,10000,1,1,80)' in script
    assert "before=fileread('/proc/stat');" in script
    assert "after=fileread('/proc/stat');" in script
    assert 'dictionary_change=max(abs(Dout(:)-D(:)))' in script
    assert 'num2hex(I(:))' in script and 'num2hex(D(:))' in script


def test_author_budget_cpu_text_preserves_busy_sibling():
    before = parse_cpu('cpu0 0 0 0 0 0 0 0 0\ncpu1 0 0 0 0 0 0 0 0\n')
    after = parse_cpu('cpu0 100 0 0 0 0 0 0 0\ncpu1 0 0 0 100 0 0 0 0\n')
    assert eligible(cpu_delta(before, after))
    after['cpu1'][0] = 10
    assert not eligible(cpu_delta(before, after))
