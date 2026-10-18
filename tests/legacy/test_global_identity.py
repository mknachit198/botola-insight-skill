"""Contrôles de l'abstention, des conflits et des frontières de propagation."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from raja_engine.global_identity import anchor_seeds,solve,split_safe_segments


def node(nid,allowed=(3,7),start=None,end=None):
    return {'id':nid,'allowed_players':list(allowed),'n_obs':10,'shot':1,
            'start':nid*2 if start is None else start,'end':nid*2+1 if end is None else end}


def edge(a,b,weight=.97):
    return {'a':a,'b':b,'continuity_weight':weight}


def anchor(nid,number):
    return {'node':nid,'number':number}


class GlobalIdentityTests(unittest.TestCase):
    def test_no_reading_means_unknown_even_with_single_allowed_player(self):
        result=solve([node(1,(3,))],[],[],[])
        self.assertEqual(result['accepted'],{})

    def test_a_number_propagates_only_along_a_motion_candidate(self):
        result=solve([node(1),node(2),node(3)],[edge(1,2)],[],[anchor(1,3)])
        self.assertEqual(result['accepted'],{1:3,2:3})

    def test_two_equally_likely_numbers_abstain_globally(self):
        result=solve([node(1),node(2),node(3)],[edge(1,3),edge(2,3)],[],[anchor(1,3),anchor(2,7)])
        self.assertNotIn(3,result['accepted'])
        self.assertEqual(result['records'][2]['reason'],'global_margin_too_small')

    def test_simultaneous_rivals_cannot_share_one_number(self):
        result=solve([node(1),node(2),node(3)],[edge(1,2),edge(1,3)],[[2,3]],[anchor(1,3)])
        self.assertEqual(result['accepted'],{1:3})
        self.assertEqual(result['simultaneity_violations'],0)

    def test_conflicting_reads_do_not_seed_neighbors(self):
        result=solve([node(1),node(2)],[edge(1,2)],[],[anchor(1,3),anchor(1,7)])
        self.assertEqual(result['accepted'],{})
        self.assertEqual(result['records'][0]['reason'],'conflicting_number_reads')

    def test_simultaneous_duplicate_reads_are_both_flagged(self):
        seeds,conflicts=anchor_seeds([anchor(1,3),anchor(2,3)],[[1,2]])
        self.assertEqual(seeds,{})
        self.assertEqual(set(conflicts),{1,2})

    def test_absent_number_cannot_seed_other_nodes(self):
        result=solve([node(1),node(2,(3,7,11))],[edge(1,2)],[],[anchor(1,11)])
        self.assertEqual(result['accepted'],{})

    def test_a_direct_contrary_number_blocks_motion_propagation(self):
        result=solve([node(1),node(2),node(3)],[edge(1,2),edge(2,3)],[],[anchor(1,3),anchor(2,7)])
        self.assertEqual(result['accepted'],{1:3,2:7,3:7})

    def test_low_weight_link_remains_unknown(self):
        result=solve([node(1),node(2)],[edge(1,2,.79)],[],[anchor(1,3)])
        self.assertEqual(result['accepted'],{1:3})

    def test_crossings_and_gaps_are_segment_boundaries(self):
        track={'obs':[{'t':t} for t in (0.,.2,.4,.6,.8,1.6,1.8)],'risk_times':[.4,.6]}
        segments=split_safe_segments(track)
        self.assertEqual([[o['t'] for o in s['obs']] for s in segments],[[0.,.2],[.4],[.6],[.8],[1.6,1.8]])
        self.assertEqual([s['crossing_observation'] for s in segments],[False,True,True,False,False])

    def test_invalid_motion_link_is_rejected(self):
        a,b=node(1),node(2)
        b['shot']=2
        with self.assertRaises(ValueError):solve([a,b],[edge(1,2)],[],[anchor(1,3)])
        b['shot']=1;b['start']=a['end']
        with self.assertRaises(ValueError):solve([a,b],[edge(1,2)],[],[anchor(1,3)])

    def test_crossing_cannot_be_a_motion_relay(self):
        a,b=node(1),node(2)
        b['crossing_observation']=True
        with self.assertRaises(ValueError):solve([a,b],[edge(1,2)],[],[anchor(1,3)])

    def test_shortest_alternative_is_proven_optimal_before_acceptance(self):
        result=solve([node(1),node(2)],[edge(1,2)],[],[anchor(1,3)])
        row=result['records'][1]
        self.assertTrue(result['optimal'])
        self.assertGreaterEqual(row['global_margin'],.25)
        self.assertFalse(result['production_approved'])
        self.assertFalse(result['weights_are_calibrated_probabilities'])


if __name__=='__main__':unittest.main()
