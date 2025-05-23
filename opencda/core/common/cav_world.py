# -*- coding: utf-8 -*-

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import importlib

from opencda.customize.core.v2x.network_manager import NetworkManager


class CavWorld(object):
    """
    A customized world object to save all CDA vehicle
    information and shared ML models. During co-simulation,
    it is also used to save the sumo-carla id mapping.

    Parameters
    ----------
    apply_ml : bool
        Whether apply ml/dl models in this simulation, please make sure
        you have install torch/sklearn before setting this to True.

    Attributes
    ----------
    vehicle_id_set : set
        A set that stores vehicle IDs.

    _vehicle_manager_dict : dict
        A dictionary that stores vehicle managers.

    _platooning_dict : dict
        A dictionary that stores platooning managers.

    _rsu_manager_dict : dict
        A dictionary that stores RSU managers.

    ml_manager : opencda object.
        The machine learning manager class.
    """
    network_manager = None
    def __init__(self, apply_ml=False,
                 apply_cp=False,
                 coperception_params=None,
                 network_params=None,
                 world_params=None):

        self.vehicle_id_set = set()
        self._vehicle_manager_dict = {}
        self._platooning_dict = {}
        self._rsu_manager_dict = {}
        self._evaluate_vehicle_manager_dict = {}
        self.ml_manager = None
        self.ego_id = None

        if apply_ml and apply_cp and coperception_params:
            ml_manager = getattr(importlib.import_module(
                "opencda.customize.ml_libs.opencood_manager"), 'OpenCOODManager')
            print("opencda.customize.ml_libs.opencood_manager", 'OpenCOODManager')
            self.ml_manager = ml_manager(coperception_params)
        elif apply_ml:
            # we import in this way so the user don't need to install ml
            # packages unless they require to
            ml_manager = getattr(importlib.import_module(
                "opencda.customize.ml_libs.ml_manager"), 'MLManager')
            print("opencda.customize.ml_libs.ml_manager", 'MLManager')
            # initialize the ml manager to load the DL/ML models into memory
            self.ml_manager = ml_manager()
        # this is used only when co-simulation activated.
        self.sumo2carla_ids = {}

        self.fixed_delta_seconds = world_params.get('fixed_delta_seconds', 0.05)
        self.frequency = 1 / self.fixed_delta_seconds

        self.network_enabled = False
        # print(network_params)
        if network_params:
            self.network_enabled = network_params['enabled']
            network_params.update({'time_slot': self.fixed_delta_seconds})
            if self.network_enabled and CavWorld.network_manager is None:
                CavWorld.network_manager = NetworkManager(cav_world=self, \
                                                            config=network_params)
        

    def update_global_ego_id(self, id=0):
        """
        Return the smallest id as the ego_id
        """
        self.ego_id = min(self.vehicle_id_set) if len(self.vehicle_id_set) > 0 else id
        #print('ego_id:', self.ego_id)
    
    def get_ego_vehicle_manager(self):
        for vid, vm in self.get_evaluate_vehicle_managers().items():
            #print('vid:', vid, self.ego_id)
            if vid == self.ego_id:
                #print('find')
                return vm
        return None

    def update_vehicle_manager(self, vehicle_manager, isTrafficVehicle):
        """
        Update created CAV manager to the world.

        Parameters
        ----------
        vehicle_manager : opencda object
            The vehicle manager class.
        """
        # self._vehicle_manager_dict.update(
        #     {vehicle_manager.vid: vehicle_manager})
        # if not isTrafficVehicle:
        #     self._evaluate_vehicle_manager_dict.update(
        #     {vehicle_manager.vid: vehicle_manager})
        vid = vehicle_manager.vehicle.id
        self.vehicle_id_set.add(vid)
        self._vehicle_manager_dict.update(
            {vid: vehicle_manager})
        if not isTrafficVehicle:
            self._evaluate_vehicle_manager_dict.update(
            {vid: vehicle_manager})

        self.update_global_ego_id()

    def update_platooning(self, platooning_manger):
        """
        Add created platooning.

        Parameters
        ----------
        platooning_manger : opencda object
            The platooning manager class.
        """
        self._platooning_dict.update(
            {platooning_manger.pmid: platooning_manger})

    def update_rsu_manager(self, rsu_manager):
        """
        Add rsu manager.

        Parameters
        ----------
        rsu_manager : opencda object
            The RSU manager class.
        """
        self._rsu_manager_dict.update({rsu_manager.rid: rsu_manager})

    def update_sumo_vehicles(self, sumo2carla_ids):
        """
        Update the sumo carla mapping dict. This is only called
        when cosimulation is conducted.

        Parameters
        ----------
        sumo2carla_ids : dict
            Key is sumo id and value is carla id.
        """
        self.sumo2carla_ids = sumo2carla_ids

    def get_vehicle_managers(self):
        """
        Return vehicle manager dictionary.
        """
        return self._vehicle_manager_dict
    
    def get_vehicle_manager(self, vehicle_id):
        return self._vehicle_manager_dict[vehicle_id]
    
    def get_evaluate_vehicle_managers(self):
        """
        Return vehicle manager dictionary for evaluation.
        """
        return self._evaluate_vehicle_manager_dict

    def get_platoon_dict(self):
        """
        Return existing platoons.
        """
        return self._platooning_dict

    def locate_vehicle_manager(self, loc):
        """
        Locate the vehicle manager based on the given location.

        Parameters
        ----------
        loc : carla.Location
            Vehicle location.

        Returns
        -------
        target_vm : opencda object
            The vehicle manager at the give location.
        """

        target_vm = None
        for key, vm in self._vehicle_manager_dict.items():
            x = vm.localizer.get_ego_pos().location.x
            y = vm.localizer.get_ego_pos().location.y

            if loc.x == x and loc.y == y:
                target_vm = vm
                break

        return target_vm

    def destroy(self):
        for vehicle_manager in self._vehicle_manager_dict.values():
            vehicle_manager.destroy()
        for rsu_manager in self._rsu_manager_dict.values():
            rsu_manager.destory()

    # def __del__(self):
    #     try:
    #         self.destroy()
    #     except Exception as e:
    #         print(f'destroying {self.__class__}: {e}')