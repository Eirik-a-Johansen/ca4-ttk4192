        # 5.0) Testing the GNC module (uncomment lines to test)

        # aea
        # aea 2
        # aea 3
        # aea 4
        

		# 5.1) Starting the AI Planner
       
        # a_plan=run_stp_planner("/home/appuser/catkin_ws/src/temporal-planning-main/temporal-planning/domains/ttk4192/domain/PDDL_domain_1_improved.pddl",
        #                  "/home/appuser/catkin_ws/src/temporal-planning-main/temporal-planning/domains/ttk4192/problem/PDDL_problem_1.pddl")    
        # if a_plan:
        #    print(" ---Executing Graph planner --- ")
        #    time.sleep(1)
        #    if len(sys.argv) != 1 and len(sys.argv) != 3:
        #         print("Usage: GraphPlan.py domainName problemName")
        #         exit()
        #     # Here you need to load your PDDL domain
        #    dir_p_dom="/home/appuser/catkin_ws/src/temporal-planning-main/temporal-planning/domains/ttk4192/domain/"
        #    dir_p_prob="/home/appuser/catkin_ws/src/temporal-planning-main/temporal-planning/domains/ttk4192/problem/"

        #    domain = dir_p_dom+"PDDL_domain_1_improved.pddl"
        #    problem = dir_p_prob+"PDDL_problem_1.pddl"
        #    if len(sys.argv) == 3:
        #         domain = str(sys.argv[1])
        #         problem = str(sys.argv[2])
        #    gp = GraphPlan(domain, problem)
        #    start = time.time()
        #    plan = gp.graphPlan()
        #    elapsed = time.time() - start
        #    plan=np.array(plan)
        #    l=[]
        #     #print([plan.action for action in plan])
        #    if plan is not None:
        #         print("Plan found with %d actions in %.2f seconds" %
        #             (len([act for act in plan if not act.isNoOp()]), elapsed))            
        #         for i in range(len(plan)):
        #             #print(plan[i])
        #             l.append(plan[i])
        #    else:
        #         print("Could not find a plan in %.2f seconds" % elapsed)

        #     #print(l[1])
        #    m=[]
        #    for i in range(len(l)):
        #         a=str(l[i])
        #         for k in a:
        #             if a[0].isupper():
        #                 m.append(a)
        #                 break
        #    plan_general=m
        #    #print("Plan in graph -plan",plan_general)
        #    # expansion of names of actions graph notation
        #    for i in range(len(plan_general)):
        #        if plan_general[i]=="Pr2":
        #            plan_general[i]="taking_photo"
        #        if plan_general[i]=="Tr3":
        #            plan_general[i]="making_turn"
        #    print("Plan: ",plan_general)
        # else:
        #    time.sleep()
        #    print("No valid option")
   
    
        # # 5.2) Reading the plan 
        # print("  ")
        # print("Reading the plan from AI planner")
        # print("  ")
        # plan_general=plan_general
        # print(plan_general[0])

        # # 5.3) Start mission execution 
        # # convert string into functions and executing
        # print("")
        # print("Starting mission execution")
        # # Start simulations with battery = 100%
        # battery=100
        # task_finished=0
        # task_total=len(plan_general)
        # i_ini=0
        # while i_ini < task_total:
        #     move_robot_waypoint0_waypoint1()
        #     #taking_photo_exe()

        #     plan_temp=plan_general[i_ini].split()
        #     print(plan_temp)
        #     if plan_temp[0]=="check_pump_picture_ir":
        #         print("Inspect -pump")
        #         time.sleep(1)

        #     if plan_temp[0]=="check_seals_valve_picture_eo":
        #         print("check-valve-EO")

        #         time.sleep(1)

        #     if plan_temp[0]=="move_robot":
        #         print("move_robot_waypoints")

        #         time.sleep(1)

        #     if plan_temp[0]=="move_charge_robot":
        #         print("")
        #         print("Going to rechard robot")

        #         time.sleep(1)

        #     if plan_temp[0]=="charge_battery":
        #         print(" ")
        #         print("charging battery")

        #         time.sleep(1)


        #     i_ini=i_ini+1  # Next tasks


        # print("")
        # print("--------------------------------------")
        # print("All tasks were performed successfully")
        # time.sleep(10) 