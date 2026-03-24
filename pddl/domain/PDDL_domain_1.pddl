(define (domain robplan)
    (:requirements :typing :durative-actions :strips :fluents)

    (:types  
     turtlebot robot camera camera_eo camera_ir robo_arm charger - vehicle
     vehicle photo valve pump pipe sound gas_ind obj battery - subject
     city_location city - location
     waypoint battery_station - city_location
     route
    )

    (:predicates
        (at ?physical_obj1 - subject ?location1 - location)
        (available ?vehicle1 - vehicle)
        (available ?camera1 - camera)
        (connects ?route1 - route ?location1 - location ?location2 - location)
        (in_city ?location1 - location ?city1 - city)
        (route_available ?route1 - route)
        (no_photo ?subject1 - subject)
        (photo ?subject1 - subject)
        (no_seals_check ?subject1 - subject)
        (seals_check ?subject1 - subject)
        (battery_low ?vehicle1 - vehicle)
        (battery_charged ?vehicle1 - vehicle)
        (no_check_pump ?subject1 - subject)
        (check_pump ?subject1 - subject)
    )

    (:functions
        (distance ?O - location ?L - location)
        (route-length ?O - route)
        (speed ?V - vehicle)
    )

    ; --- Action 1: Move robot between waypoints ---
    (:durative-action move_robot
        :parameters (?V - robot ?O - location ?L - location ?R - route)
        :duration (= ?duration (/ (route-length ?R) (speed ?V)))
        :condition (and
            (at start (at ?V ?O))
            (at start (connects ?R ?O ?L))
        )
        :effect (and
            (at start (not (at ?V ?O)))
            (at end (at ?V ?L))
        )
    )

    ; --- Action 2: Inspect valve with EO camera ---
    (:durative-action check_seals_valve_picture_EO
        :parameters (?V - robot ?L - location ?G - camera_eo ?B - valve)
        :duration (= ?duration 10)
        :condition (and
            (over all (at ?V ?L))
            (at start (at ?B ?L))
            (at start (available ?G))
            (at start (no_seals_check ?B))
        )

        :effect (and
            (at start (not (no_seals_check ?B)))
            (at end (seals_check ?B))
        )
    )

    ; --- Action 3: Take picture of pump with IR camera ---
    (:durative-action take_picture_pump_IR
        :parameters (?V - robot ?L - location ?G - camera_ir ?P - pump)
        :duration (= ?duration 10)
        :condition (and
            (over all (at ?V ?L))
            (at start (at ?P ?L))
            (at start (available ?G))
            (at start (no_check_pump ?P))
        )
        :effect (and
            (at start (not (no_check_pump ?P)))
            (at end (check_pump ?P))
        )
    )

    ; --- Action 4: Charge battery at charging station ---
    (:durative-action charge_battery
        :parameters (?V - robot ?L - battery_station ?C - charger)
        :duration (= ?duration 20)
        :condition (and
            (over all (at ?V ?L))
            (at start (at ?C ?L))
            (at start (battery_low ?V))
        )
        :effect (and
            (at start (not (battery_low ?V)))
            (at end (battery_charged ?V))
        )
    )
)