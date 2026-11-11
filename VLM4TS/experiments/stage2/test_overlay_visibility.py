import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt
from run_stage2_review_hfz_v2 import render


class OverlayTests(unittest.TestCase):
    def test_identical_reference_drawn_above_candidate_as_dashed_line(self):
        train=np.zeros((300,1));test=np.zeros((224,1));test[110,0]=1.
        class Provider:
            def band(self,train,tile,channel,focus,lo,hi,**kwargs):
                fs,fe=focus;values=tile[fs:fe,channel];return (np.stack([values,values,values]),dict(period=None,context=48,n_refs=8,limited=False))
        item=dict(start=0,end=224,machine='synthetic',band='candidate',channels=[dict(channel=0,zoom_center=110,dominant_component='test')],show_event_core=False)
        with tempfile.TemporaryDirectory() as tmp,patch.object(plt,'close'):
            render(train,test,item,Path(tmp)/'image.png',reference_provider=Provider(),display_reference_quality=False)
            fig=plt.gcf();zoom=fig.axes[-1];lines={line.get_label():line for line in zoom.lines}
            self.assertGreater(lines['normal shape (median)'].get_zorder(),lines['candidate'].get_zorder())
            self.assertEqual(lines['normal shape (median)'].get_linestyle(),'--')
            np.testing.assert_allclose(lines['normal shape (median)'].get_ydata(),lines['candidate'].get_ydata())
        plt.close('all')

if __name__=='__main__':unittest.main()
