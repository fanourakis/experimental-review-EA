import argparse
import sys
import time

from openea.modules.args.args_hander import check_args, load_args
from openea.modules.load.kgs import read_kgs_from_folder
from openea.models.semantic import RotatE
from openea.approaches import MTransE
from openea.approaches import GCN_Align
from openea.approaches import MultiKE
from openea.approaches import KDCoE
from openea.approaches import RDGCN
from openea.models.basic_model import BasicModel


class ModelFamily(object):
    BasicModel = BasicModel


    RotatE = RotatE

   
    MTransE = MTransE
    
    GCN_Align = GCN_Align
    KDCoE = KDCoE

    
    MultiKE = MultiKE

    RDGCN = RDGCN



def get_model(model_name):
    return getattr(ModelFamily, model_name)


if __name__ == '__main__':
    t = time.time()
    args = load_args(sys.argv[1])
    args.training_data = args.training_data + sys.argv[2] + '/'
    args.dataset_division = sys.argv[3]
    print(args.embedding_module)
    print(args)
    remove_unlinked = False
    if args.embedding_module == "RSN4EA":
        remove_unlinked = True
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                               remove_unlinked=remove_unlinked)
    model = get_model(args.embedding_module)()
    model.set_args(args)
    model.set_kgs(kgs)
    model.init()
    model.run()
    model.test()
    model.save()
    print("Total run time = {:.3f} s.".format(time.time() - t))