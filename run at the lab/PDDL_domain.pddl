(define (domain robplan)
(:requirements :typing :durative-actions :strips :fluents)
    (:types
        vehicle tools asset - subject
        robot - vehicle
        camera robo_arm charger - tools
        camera_eo camera_ir - camera
        valve pump pipe sound gas_ind obj battery - asset
        city_location city - location
        waypoint - city_location
        route
    )

  (:predicates
    (at ?physical_obj1 - subject ?location1 - location)
    (available ?x - subject)   
    (connects ?route1 - route ?location1 - location ?location2 - location)
    (route_available ?route1 - route)
    (no_photo ?subject1 - subject)
    (photo ?subject1 - subject)
    (no_seals_check ?subject1 - subject)
    (seals_check ?subject1 - subject)
    (free-charger ?c - charger)
    (charged ?v - robot)
    (needs-charge ?v - robot)
    (no_grasped ?o - asset)
    (grasped ?o - asset)


   )

    (:functions 
            (distance ?O - location ?L - location)
            (route-length ?O - route)
	        (speed ?V - vehicle)
    )
    



  (:durative-action move_robot
       :parameters ( ?V - robot ?O - location ?L - location ?R - route)
       :duration (= ?duration (/ (route-length ?R) (speed ?V)))
       :condition (and 
			(at start (at ?V ?O))
            (at start (connects ?R ?O ?L))
            (at start (charged ?V))
       )
       :effect (and 
		  (at start (not (at ?V ?O)))
                  (at end (at ?V ?L))
        )
    )


 (:durative-action check_seals_valve_picture_EO
       :parameters ( ?V - robot ?L - location ?G - camera_eo ?B - valve)
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

 (:durative-action take_picture_pump_ir
        :parameters (?V - robot ?L - location ?G - camera_ir ?P - pump)
        :duration (= ?duration 10)
        :condition (and
            (over all (at ?V ?L))
            (at start (at ?P ?L))
            (at start (available ?G))
            (at start (no_photo ?P))
            )
        :effect (and
        (at start (not (no_photo ?P)))
        (at end (photo ?P))
        ) 
    )

    (:durative-action grasp_object
        :parameters (?V - robot ?L - location ?A - robo_arm ?O - asset)
        :duration (= ?duration 3)
        :condition (and
            (over all (at ?V ?L))
            (at start (at ?O ?L))
            (at start (available ?A))
            (at start (no_grasped ?O))
        )
        :effect (and
            (at start (not (no_grasped ?O)))
            (at end (grasped ?O))
        )
    )

    (:durative-action charge_battery
        :parameters (?V - robot ?L - location ?C - charger)
        :duration (= ?duration 10)
        :condition (and
            (over all (at ?V ?L))
            (at start (at ?C ?L))
            (at start (free-charger ?C))
            (at start (needs-charge ?V))
        )
        :effect (and
            (at start (not (free-charger ?C)))
            (at start (not (needs-charge ?V)))
            (at end (charged ?V))
            (at end (free-charger ?C))
        )
    )
)