import abc


class PlayerABC(abc.ABCMeta):
    """ PlayerABC (Player for Abstract Basic Class)  """

    def get_action(self,):
        raise NotImplementedError()
