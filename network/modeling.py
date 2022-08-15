from .mtvc import EmbeddingResNet, EmbeddingNet, SiameseNet, TripletNet, ClassificationNet


def _load_ClasModel(arch_type, embedding_name, num_classes):
    """
    村落分类模型加载
    """
    # create embedding_net
    if embedding_name == 'embeddingNet':
        embedding_net = EmbeddingNet()
    elif embedding_name == 'embeddingResNet':
        embedding_net = EmbeddingResNet()
    else:
        raise ValueError("embedding_name error!Please check and try again!")
    # create model
    if arch_type == 'classificationNet':
        model = ClassificationNet(embedding_net, num_classes)
    elif arch_type == 'siameseNetwork':
        model = SiameseNet(embedding_net)
    elif arch_type == 'tripletNetwork':
        model = TripletNet(embedding_net)
    elif arch_type == 'onlinePairSelection':
        model = embedding_net
    elif arch_type == 'onlineTripletSelection':
        model = embedding_net
    else:
        raise ValueError("arch_type error!Please check and try again!")
    return model


# Classification model
# Mtvc Baseline: classification with softmax
def classificationNet(embedding_name, num_classes=6):
    """Constructs a Mtvc classification with softmax"""
    return _load_ClasModel('classificationNet', embedding_name, num_classes=num_classes)


# siamese
def siameseNetwork(embedding_name, num_classes=6):
    """Constructs a Mtvc model with embedding Network"""
    return _load_ClasModel('siameseNetwork', embedding_name, num_classes=num_classes)


# triplet
def tripletNetwork(embedding_name, num_classes=6):
    """Constructs a Mtvc model with embedding Network"""
    return _load_ClasModel('tripletNetwork', embedding_name, num_classes=num_classes)


# onlinePairSelection
def onlinePairSelection(embedding_name, num_classes=6):
    """Constructs a Mtvc model with embedding Network"""
    return _load_ClasModel('onlinePairSelection', embedding_name, num_classes=num_classes)


# onlineTripletSelection
def onlineTripletSelection(embedding_name, num_classes=6):
    """Constructs a Mtvc model with embedding Network"""
    return _load_ClasModel('onlineTripletSelection', embedding_name, num_classes=num_classes)
