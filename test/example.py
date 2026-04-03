import carla
import random
import time

def main():
    # 1. 连接到Carla服务器
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # 2. 加载 Town05
    world = client.load_world('Town05')
    weather = world.get_weather()
    weather.sun_altitude_angle = 90.0 
    world.set_weather(weather)

    # 3. 获取蓝图库与生成点
    blueprint_library = world.get_blueprint_library()
    # spawn_points = world.get_map().get_spawn_points()
    map = world.get_map()
    # 控制要生成的车辆数量
    num_vehicles = 50

    x_min, x_max = int(-180), int(-80)
    y_min, y_max = int(-60), int(60)
    spawn_set = set()
    for x in range(x_min, x_max, 4):
        for y in range(y_min, y_max, 4):
            print(x, y)
            location = carla.Location(x=x, y=y, z=0.3)
            way_point = map.get_waypoint(location).transform

            spawn_set.add((way_point.location.x,
                            way_point.location.y,
                            way_point.location.z,
                            way_point.rotation.roll,
                            way_point.rotation.yaw,
                            way_point.rotation.pitch))

    spawn_points = [carla.Transform(carla.Location(x=coordinates[0],
                                                y=coordinates[1],
                                                z=coordinates[2] + 0.3),
                                    carla.Rotation(
                                    roll=coordinates[3],
                                    yaw=coordinates[4],
                                    pitch=coordinates[5])) 
                    for coordinates in spawn_set]

    # 4. 选择车辆蓝图
    # vehicle_blueprints = blueprint_library.filter('vehicle.*')
    vehicle_blueprints = [
        blueprint_library.find('vehicle.audi.a2'),
        blueprint_library.find('vehicle.audi.tt'),
        blueprint_library.find('vehicle.dodge_charger.police'),
        blueprint_library.find('vehicle.jeep.wrangler_rubicon'),
        blueprint_library.find('vehicle.chevrolet.impala'),
        blueprint_library.find('vehicle.mini.cooperst'),
        blueprint_library.find('vehicle.audi.etron'),
        blueprint_library.find('vehicle.mercedes-benz.coupe'),
        blueprint_library.find('vehicle.bmw.grandtourer'),
        blueprint_library.find('vehicle.toyota.prius'),
        blueprint_library.find('vehicle.citroen.c3'),
        blueprint_library.find('vehicle.mustang.mustang'),
        blueprint_library.find('vehicle.tesla.model3'),
        blueprint_library.find('vehicle.lincoln.mkz2017'),
        blueprint_library.find('vehicle.seat.leon'),
        blueprint_library.find('vehicle.nissan.patrol'),
        blueprint_library.find('vehicle.nissan.micra'),
    ]

    vehicles_list = []

    # 5. 生成车辆
    random.shuffle(spawn_points)
    # print(spawn_list)
    n = 0
    for transform in spawn_points:
        if n >= num_vehicles:
            break
        blueprint = random.choice(vehicle_blueprints)

        # 随机车辆颜色与车牌
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        blueprint.set_attribute('role_name', 'autopilot')

        # 生成车辆
        vehicle = world.try_spawn_actor(blueprint, transform)
        if not vehicle:
            continue
        vehicles_list.append(vehicle)
        print(f"Spawned vehicle {n+1}/{num_vehicles} id={vehicle.id}")
        n += 1

    # 6. 启动同步模式可选（提高稳定性）
    settings = world.get_settings()
    settings.synchronous_mode = False  # 若已在同步模式，可设为True
    world.apply_settings(settings)

    # 7. 启动车辆自动驾驶（利用TrafficManager）
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(False)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)

    # for vehicle in vehicles_list:
    #     vehicle.set_autopilot(True, traffic_manager.get_port())

    # print("所有车辆已经启动自动驾驶。")

    # 8. 运行一段时间
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("用户中断。销毁所有车辆。")

    # 9. 清理
    print('销毁车辆...')
    client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
    print('完成！')

if __name__ == '__main__':
    main()
