from src.Ensemble import Ensemble, EnsembleMethods
from src.ModelBuilder import get_Encoder, get_FCN, get_MCDCNN, get_MLP, get_Resnet, get_Time_CNN


def build_ensemble(model_builder, input_size: int, output_size: int, ensemble_type=EnsembleMethods.AVERAGE) -> Ensemble:
    return Ensemble(models=(list(map(lambda f: f(input_size=input_size, output_size=output_size),
                                     model_builder))),
                    ensemble_type=ensemble_type)


def get_Resnets(input_size, output_size, ensemble_type=EnsembleMethods.AVERAGE) -> Ensemble:
    return build_ensemble([get_Resnet] * 4, input_size, output_size, ensemble_type)


def get_All(input_size, output_size, ensemble_type=EnsembleMethods.AVERAGE) -> Ensemble:
    return build_ensemble([get_Encoder, get_FCN, get_MCDCNN, get_MLP, get_Resnet, get_Time_CNN], input_size,
                          output_size, ensemble_type)


def get_NNE(input_size, output_size, ensemble_type=EnsembleMethods.AVERAGE) -> Ensemble:
    return build_ensemble([get_MLP, get_MLP, get_MLP], input_size, output_size, ensemble_type)


"""
def build_ensemble(model_builder, input_size: int, output_size: int, ensemble_type=EnsembleMethods.AVERAGE) -> Ensemble:
    models = []
    for builder in model_builder:
        models.append(builder(input_size, output_size))
    return Ensemble(models=models, ensemble_type=ensemble_type)
from functools import partial
get_Resnets = partial(build_ensemble, [get_Resnet] * 10)
get_All = partial(build_ensemble, [get_Encoder, get_FCN, get_MCDCNN, get_MLP, get_Resnet, get_Time_CNN])
get_NNE = partial(build_ensemble, [get_FCN, get_Resnet, get_Encoder])
"""
