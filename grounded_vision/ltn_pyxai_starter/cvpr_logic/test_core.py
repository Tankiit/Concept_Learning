import itertools
import unittest
import numpy as np
import torch
from .model import LogicLayer, LogicHead
from .verify import Explainer,predict,calibrate,allowed_values,crosscheck_pyxai
from .data import demo,validate


class Tests(unittest.TestCase):
    def test_gate_order_and_relaxation(self):
        layer=LogicLayer(2,16)
        layer.wiring[:]=torch.tensor([[0]*16,[1]*16])
        layer.weights.data.fill_(-20)
        layer.weights.data[torch.arange(16),torch.arange(16)]=20
        x=torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
        layer.eval()
        want=torch.tensor([[(g>>(3-j))&1 for g in range(16)] for j in range(4)],dtype=torch.float32)
        torch.testing.assert_close(layer(x),want)
        layer.train(); soft=torch.rand(9,2,requires_grad=True)
        loss=layer(soft).square().mean(); loss.backward()
        self.assertGreater(float(soft.grad.norm()),0)
        self.assertGreater(float(layer.weights.grad.norm()),0)

    def test_sat_matches_exhaustive_multiclass_and_ties(self):
        # Exercise all partial assignments, random circuits and deliberate ties.
        for seed in [4,9,15]:
            torch.manual_seed(seed)
            head=LogicHead(4,3,width=6,depth=2)
            if seed==15: head.layers[-1].weights.data.zero_() # all false => class 0
            circuit=head.export(['a','b','c','d'])
            worlds=np.array(list(itertools.product([0,1],repeat=4)))
            labels=predict(circuit,worlds)
            head.eval()
            with torch.no_grad(): actual=head(torch.tensor(worlds,dtype=torch.float32)).argmax(1).numpy()
            np.testing.assert_array_equal(actual,labels)
            exp=Explainer(circuit)
            try:
                for partial in itertools.product([-1,0,1],repeat=4):
                    reason=tuple(j+1 if v else -(j+1) for j,v in enumerate(partial) if v!=-1)
                    mask=np.ones(len(worlds),dtype=bool)
                    for j,v in enumerate(partial):
                        if v!=-1: mask &= worlds[:,j]==v
                    for target in range(3):
                        self.assertEqual(exp.sufficient(reason,target),bool(np.all(labels[mask]==target)))
            finally: exp.close()

    def test_supported_reason_and_empty_set(self):
        # y=1 iff a OR b; class0 contributes FALSE, class1 contributes OR.
        circuit={'concept_names':['a','b'],'classes':2,'layers':[{'a':[0,0],'b':[1,1],'gate':[0,7]}],
                 'aggregation':'equal_group_sum','tie_break':'lowest_class_index','tau':1.}
        exp=Explainer(circuit)
        try:
            r,s=exp.shortest([1,1],np.array([[1,1],[0,1]],bool))
            self.assertEqual(r,(2,)); self.assertEqual(s,'verified')
            self.assertIsNone(exp.shortest([1,1],np.array([[0,0],[0,1]],bool))[0])
            self.assertIsNone(exp.shortest([1,1],np.ones((2,2),bool))[0])
            self.assertEqual(exp.shortest([1,1],max_candidates=1)[1],'candidate_budget')
            self.assertEqual(crosscheck_pyxai(circuit)['worlds'],4)
            constant=dict(circuit)
            constant['layers']=[{'a':[0,0],'b':[1,1],'gate':[0,0]}]
            self.assertEqual(crosscheck_pyxai(constant)['worlds'],4)
        finally: exp.close()

    def test_joint_calibration_and_data_contract(self):
        q=np.array([[.9,.1],[.6,.7],[.8,.4]])
        c=np.array([[1,0],[1,1],[0,0]])
        self.assertAlmostEqual(calibrate(q,c,.5),.4)
        self.assertEqual(calibrate(q,c,.01),1.)
        self.assertTrue(allowed_values([.5,.5],1).all())
        d=demo(counts=(8,8,8,8),features=True)
        bad=dict(d); bad['observed']=d['observed'].copy(); bad['observed'][-1,0]=False
        with self.assertRaises(ValueError): validate(bad)
        bad=dict(d); bad['groups']=d['groups'].copy(); bad['groups'][-1]=bad['groups'][0]
        with self.assertRaises(ValueError): validate(bad)

    def test_checkpoint_preserves_wiring(self):
        torch.manual_seed(3); h=LogicHead(4,2,width=8)
        state=h.state_dict(); circuit=h.export(['a','b','c','d'])
        torch.manual_seed(22); other=LogicHead(4,2,width=8)
        other.load_state_dict(state)
        self.assertEqual(circuit,other.export(['a','b','c','d']))


if __name__=='__main__': unittest.main()
