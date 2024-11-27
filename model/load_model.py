from model.models import *
from model.models_g import KHWl_HGNN, one_HWL, CP_HGNN, UniGCNII_g, AllSetTransformer_g, EquivSetGNN_g, AllDeepSet_g, \
    KHWl_HGNN_trans, KHWl_HGNN_Deep, MLP_model_g


def parse_method(args):
    if args.task_kind in ['node_classification', 'link_prediction']:
        if args.model_name == 'AllDeepSet':
            model = AllDeepSet(args)
        elif args.model_name == 'AllSetTransformer':
            model = AllSetTransformer(args)
        elif args.model_name == 'UniGCNII':
            model = UniGCNII(args)
        elif args.model_name =='EnUniGCNII':
            model = EnUniGCNII()
        elif args.model_name == 'MLP':
            model = MLP_model(args)
        elif args.model_name == 'EDHNN':
            model = EquivSetGNN(args)
        elif args.model_name == 'MultiHeadHNN':
            model = Str_EquivSetGNN(args)
    elif args.task_kind == 'graph_classification':
        if args.model_name == 'KhwlHGNN':
            model = KHWl_HGNN(args)
        elif args.model_name == 'KhwlTrans':
            model = KHWl_HGNN_trans(args)
        elif args.model_name == 'KhwlDeep':
            model = KHWl_HGNN_Deep(args)
        elif args.model_name == 'oneHWL':
            model = one_HWL(args)
        elif args.model_name == 'CPHGNN':
            model = CP_HGNN(args)
        elif args.model_name == 'UniGCNII':
            model = UniGCNII_g(args)
        elif args.model_name == 'AllSetTransformer':
            model = AllSetTransformer_g(args)
        elif args.model_name == 'EDHNN':
            model = EquivSetGNN_g(args)
        elif args.model_name == 'AllDeepSet':
            model = AllDeepSet_g(args)
        elif args.model_name == 'MLP':
            model = MLP_model_g(args)
    return model
