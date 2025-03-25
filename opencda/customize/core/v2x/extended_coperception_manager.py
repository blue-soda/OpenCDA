from opencda.core.sensing.perception.coperception_manager import CoperceptionManager
class ExtendedCoperceptionManager(CoperceptionManager):
    def __init__(self, vid, v2x_manager, coperception_libs):
        super(ExtendedCoperceptionManager, self). __init__(vid, v2x_manager, coperception_libs)

    def communicate(self):
        data = {}
        if self.v2x_manager is not None:
            for vid, vm in self.v2x_manager.cav_nearby.items():
                data.update({str(vid): vm})
        return data