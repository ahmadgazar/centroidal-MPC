import numpy as np
import crocoddyl
import pinocchio 

class WholeBodyModel:
    def __init__(self, conf, TRACK_CENTROIDAL):
        self.TRACK_CENTROIDAL = TRACK_CENTROIDAL
        self.dt = conf.dt
        self.dt_ctrl = conf.dt_ctrl
        self.robot_name = conf.robot_name
        # self.q0 = conf.q0
        self.rmodel = conf.rmodel
        self.rdata = conf.rmodel.createData()
        self.ee_frame_names = conf.ee_frame_names 
        self.gait = conf.gait
        self.gait_templates = conf.gait_templates 
        self.task_weights = conf.whole_body_task_weights
        # Defining the friction coefficient and normal
        self.mu = conf.mu
        self.N = conf.N
        self.Rsurf = np.eye(3)
        self.__initialize_robot(conf.q0)
        self.__set_contact_frame_names_and_indices()
        self.__load_centroidal_tracking_traj()

    def __set_contact_frame_names_and_indices(self):
        ee_frame_names = self.ee_frame_names
        if self.robot_name == 'solo12':
            self.lfFootId = self.rmodel.getFrameId(ee_frame_names[0])
            self.rfFootId = self.rmodel.getFrameId(ee_frame_names[1])
            self.lhFootId = self.rmodel.getFrameId(ee_frame_names[2])
            self.rhFootId = self.rmodel.getFrameId(ee_frame_names[3])

    def __initialize_robot(self, q0):
        #self.q0 = self.rmodel.referenceConfigurations['standing'].copy() 
        self.rmodel.defaultState = np.concatenate([q0, np.zeros(self.rmodel.nv)])
        # create croccodyl state and controls
        self.state = crocoddyl.StateMultibody(self.rmodel)
        self.actuation = crocoddyl.ActuationModelFloatingBase(self.state)

    def __load_centroidal_tracking_traj(self):
        if self.TRACK_CENTROIDAL:
            self.X = np.load('centroidal_to_wholeBody_traj.npz')['X']
            self.U = np.load('centroidal_to_wholeBody_traj.npz')['U']

    def add_swing_feet_tracking_costs(self, cost, swing_feet_tasks):
        weight = self.task_weights['footTrack']['swing']
        for task in swing_feet_tasks:
            frame_position_residual = crocoddyl.ResidualModelFrameTranslation(self.state,
                                         task[0], task[1].translation, self.actuation.nu)
            foot_track = crocoddyl.CostModelResidual(self.state, frame_position_residual)
            cost.addCost(self.rmodel.frames[task[0]].name + "_footTrack", foot_track, weight)
    
    def add_swing_feet_impact_costs(self, cost, swing_feet_tasks):
        nu = self.actuation.nu
        foot_pos_weight = self.task_weights['footTrack']['impact']
        foot_impact_weight = self.task_weights['impulseVel']
        for i in swing_feet_tasks:
            frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(self.state, i[0], i[1].translation, nu)
            frameVelocityResidual = crocoddyl.ResidualModelFrameVelocity(self.state, i[0], pinocchio.Motion.Zero(),
                                                                                              pinocchio.LOCAL, nu)
            footTrack = crocoddyl.CostModelResidual(self.state, frameTranslationResidual)
            impulseFootVelCost = crocoddyl.CostModelResidual(self.state, frameVelocityResidual)
            cost.addCost(self.rmodel.frames[i[0]].name + "_footTrack", footTrack, foot_pos_weight)
            cost.addCost(self.rmodel.frames[i[0]].name + "_impulseVel", impulseFootVelCost, foot_impact_weight)

    def add_support_contact_costs(self, contact_model, cost, support_feet_ids, forceTask):
        state, nu = self.state, self.actuation.nu
        frictionConeWeight = self.task_weights['frictionCone']
        forceTrackWeight = self.task_weights['contactForceTrack']
        if self.robot_name == 'solo12':
            nu_contact = 3
        for frame_idx in support_feet_ids: 
            support_contact = crocoddyl.ContactModel3D(self.state, frame_idx, np.array([0., 0., 0.]), nu,
                                                                                    np.array([0., 50.]))
            contact_model.addContact(self.rmodel.frames[frame_idx].name + "_contact", support_contact)
            cone = crocoddyl.FrictionCone(self.Rsurf, self.mu, 4, True)
            cone_residual = crocoddyl.ResidualModelContactFrictionCone(state, frame_idx, cone, nu)
            cone_activation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(cone.lb, cone.ub))
            friction_cone = crocoddyl.CostModelResidual(state, cone_activation, cone_residual)
            cost.addCost(self.rmodel.frames[frame_idx].name + "_frictionCone", friction_cone, frictionConeWeight)
            if forceTask is not None:
                if frame_idx == self.rfFootId:
                    spatial_force_des = pinocchio.Force(forceTask[0:3], np.zeros(3))
                if frame_idx == self.lfFootId:
                    spatial_force_des = pinocchio.Force(forceTask[3:6], np.zeros(3)) 
                if frame_idx == self.rhFootId:
                    spatial_force_des = pinocchio.Force(forceTask[6:9], np.zeros(3)) 
                if frame_idx == self.lhFootId:
                    spatial_force_des = pinocchio.Force(forceTask[9:12], np.zeros(3)) 
                force_residual = crocoddyl.ResidualModelContactForce(state, frame_idx, spatial_force_des, nu_contact, nu)
                force_track = crocoddyl.CostModelResidual(state, force_residual)
                cost.addCost(self.rmodel.frames[frame_idx].name +"contactForceTrack", force_track, forceTrackWeight)

    def add_com_position_tracking_cost(self, cost, com_des):    
        com_residual = crocoddyl.ResidualModelCoMPosition(self.state, com_des, self.actuation.nu)
        com_track = crocoddyl.CostModelResidual(self.state, com_residual)
        cost.addCost("comTrack", com_track, self.task_weights['comTrack'])
    
    def add_centroidal_momentum_tracking_cost(self, cost, hg_des):
        state, nu = self.state, self.actuation.nu
        weight = self.task_weights['centroidalTrack']
        hg_residual = crocoddyl.ResidualModelCentroidalMomentum(state, hg_des, nu)
        hg_track = crocoddyl.CostModelResidual(state, hg_residual)
        cost.addCost("centroidalTrack", hg_track, weight)        

    def add_stat_ctrl_reg_costs(self, cost, preImpact):
        nu = self.actuation.nu 
        stateWeights = np.array([0.]*3 + [500.]*3 + [0.01]*(self.rmodel.nv - 6) + [10.] * 6 + [1.]*(self.rmodel.nv - 6))
        if preImpact:
            state_reg_weight, control_reg_weight = self.task_weights['stateReg']['impact'], self.task_weights['ctrlReg']['impact']
        else:
            state_reg_weight, control_reg_weight = self.task_weights['stateReg']['stance'], self.task_weights['ctrlReg']['stance']
   
        state_bounds_weight = self.task_weights['stateBounds']
        # state regularization cost
        stateResidual = crocoddyl.ResidualModelState(self.state, self.rmodel.defaultState, nu)
        stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
        stateReg = crocoddyl.CostModelResidual(self.state, stateActivation, stateResidual)
        cost.addCost("stateReg", stateReg, state_reg_weight)
        # state bounds cost
        lb = np.concatenate([self.state.lb[1:self.state.nv + 1], self.state.lb[-self.state.nv:]])
        ub = np.concatenate([self.state.ub[1:self.state.nv + 1], self.state.ub[-self.state.nv:]])
        stateBoundsResidual = crocoddyl.ResidualModelState(self.state, nu)
        stateBoundsActivation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(lb, ub))
        stateBounds = crocoddyl.CostModelResidual(self.state, stateBoundsActivation, stateBoundsResidual)
        cost.addCost("stateBounds", stateBounds, state_bounds_weight)
        # control regularization cost
        ctrlResidual = crocoddyl.ResidualModelControl(self.state, nu)
        ctrlReg = crocoddyl.CostModelResidual(self.state, ctrlResidual)
        cost.addCost("ctrlReg", ctrlReg, control_reg_weight)
    
    def add_terminal_costs(self, feetPos):
        supportFeetIds = [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId]
        swingFootTask = []
        for i, p in zip(supportFeetIds, feetPos):
            swingFootTask += [[i, pinocchio.SE3(np.eye(3), p)]]
        terminalCostModel = self.createSwingFootModel([self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId], 
                    swingFootTask=swingFootTask, comTask=self.X[:3, -1], centroidalTask=self.X[3:9, -1],forceTask=self.U[:, -1])                            
        return terminalCostModel                    
        
    def createTrotShootingProblem(self):
        # Compute the current foot positions
        x0 = self.rmodel.defaultState
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation 
        self.comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        self.comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        self.time_idx = 0
        # Defining the action models along the time instances
        loco3dModel = []
        for gait in self.gait_templates:
            for phase in gait:
                if phase == 'doubleSupport':
                    loco3dModel += self.createDoubleSupportFootstepModels([lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0])
                elif phase == 'rflhStep':
                    loco3dModel += self.createSingleSupportFootstepModels([rfFootPos0, lhFootPos0], [self.lfFootId, self.rhFootId],
                                                                                                    [self.rfFootId, self.lhFootId])
                elif phase == 'lfrhStep':
                    loco3dModel += self.createSingleSupportFootstepModels([lfFootPos0, rhFootPos0], [self.rfFootId, self.lhFootId], 
                                                                                                    [self.lfFootId, self.rhFootId])
        if self.TRACK_CENTROIDAL:
            loco3dModelTerminal = self.add_terminal_costs([lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0])
        else:
            loco3dModelTerminal = loco3dModel[-1]             
        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModelTerminal)
        return problem
        
    def createPaceShootingProblem(self):
        # Compute the current foot positions
        x0 = self.rmodel.defaultState
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        self.comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        self.comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        self.time_idx = 0
        # Defining the action models along the time instances
        loco3dModel = []
        for gait in self.gait_templates:
            for phase in gait:
                if phase == 'doubleSupport':
                    loco3dModel += self.createDoubleSupportFootstepModels([lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0])
                elif phase == 'rfrhStep':
                    loco3dModel += self.createSingleSupportFootstepModels([rfFootPos0, rhFootPos0], [self.lfFootId, self.lhFootId],
                                                                                                    [self.rfFootId, self.rhFootId])
                elif phase == 'lflhStep':
                    loco3dModel += self.createSingleSupportFootstepModels([lfFootPos0, lhFootPos0], [self.rfFootId, self.rhFootId], 
                                                                                                    [self.lfFootId, self.lhFootId])
        if self.TRACK_CENTROIDAL:
            loco3dModelTerminal = self.add_terminal_costs([lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0])
        else:
            loco3dModelTerminal = loco3dModel[-1]             
        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModelTerminal)
        return problem

    def createBoundShootingProblem(self):
         # Compute the current foot positions
        x0 = self.rmodel.defaultState
        q0 = x0[:self.rmodel.nq]
        pinocchio.forwardKinematics(self.rmodel, self.rdata, q0)
        pinocchio.updateFramePlacements(self.rmodel, self.rdata)
        rfFootPos0 = self.rdata.oMf[self.rfFootId].translation
        rhFootPos0 = self.rdata.oMf[self.rhFootId].translation
        lfFootPos0 = self.rdata.oMf[self.lfFootId].translation
        lhFootPos0 = self.rdata.oMf[self.lhFootId].translation
        self.comRef = (rfFootPos0 + rhFootPos0 + lfFootPos0 + lhFootPos0) / 4
        self.comRef[2] = pinocchio.centerOfMass(self.rmodel, self.rdata, q0)[2].item()
        self.time_idx = 0
        # Defining the action models along the time instances
        loco3dModel = []
        for gait in self.gait_templates:
            for phase in gait:
                if phase == 'doubleSupport':
                    loco3dModel += self.createDoubleSupportFootstepModels([lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0])
                elif phase == 'rflfStep':
                    loco3dModel += self.createSingleSupportFootstepModels([rfFootPos0, lfFootPos0], [self.rhFootId, self.lhFootId],
                                                                                                    [self.rfFootId, self.lfFootId])
                elif phase == 'rhlhStep':
                    loco3dModel += self.createSingleSupportFootstepModels([rhFootPos0, lhFootPos0], [self.rfFootId, self.lfFootId], 
                                                                                                    [self.rhFootId, self.lhFootId])
        if self.TRACK_CENTROIDAL:
            loco3dModelTerminal = self.add_terminal_costs([lfFootPos0, rfFootPos0, lhFootPos0, rhFootPos0])
        else:
            loco3dModelTerminal = loco3dModel[-1]             
        problem = crocoddyl.ShootingProblem(x0, loco3dModel, loco3dModelTerminal)
        return problem

    def createDoubleSupportFootstepModels(self, feetPos):
        supportFeetIds = [self.lfFootId, self.rfFootId, self.lhFootId, self.rhFootId]
        supportKnots = self.gait['supportKnots']
        doubleSupportModel = []
        for _ in range(supportKnots):
            swingFootTask = []
            for i, p in zip(supportFeetIds, feetPos):
                swingFootTask += [[i, pinocchio.SE3(np.eye(3), p)]]
            if self.TRACK_CENTROIDAL:
                forceTask =  self.U[:, self.time_idx]
                doubleSupportModel += [self.createSwingFootModel(supportFeetIds, swingFootTask=swingFootTask, forceTask=forceTask)]
                self.time_idx += 1
            else:
                doubleSupportModel += [self.createSwingFootModel(supportFeetIds, swingFootTask=swingFootTask)]               
        return doubleSupportModel

    def createSingleSupportFootstepModels(self, feetPos0, supportFootIds, swingFootIds):
        numLegs = len(supportFootIds) + len(swingFootIds)
        comPercentage = float(len(swingFootIds)) / numLegs
        stepLength, stepHeight = self.gait['stepLength'], self.gait['stepHeight']
        numKnots = self.gait['stepKnots'] 
        # Action models for the foot swing
        footSwingModel = []
        for k in range(numKnots):
            swingFootTask = []
            for i, p in zip(swingFootIds, feetPos0):
                # Defining a foot swing task given the step length
                # resKnot = numKnots % 2
                phKnots = numKnots / 2
                if k < phKnots:
                    dp = np.array([stepLength * (k + 1) / numKnots, 0., stepHeight * k / phKnots])
                elif k == phKnots:
                    dp = np.array([stepLength * (k + 1) / numKnots, 0., stepHeight])
                else:
                    dp = np.array(
                        [stepLength * (k + 1) / numKnots, 0., stepHeight * (1 - float(k - phKnots) / phKnots)])
                tref = p + dp
                swingFootTask += [[i, pinocchio.SE3(np.eye(3), tref)]]
            if self.TRACK_CENTROIDAL:
                comTask = self.X[:3, self.time_idx]
                centroidalTask = self.X[3::, self.time_idx]
                forceTask = self.U[:, self.time_idx]
                self.time_idx += 1 
            else:    
                comTask = np.array([stepLength * (k + 1) / numKnots, 0., 0.]) * comPercentage + self.comRef
                centroidalTask = None
                forceTask = None
            if k == numKnots-1:
                preImpact = True
            else:
                preImpact = False    
            footSwingModel += [
                self.createSwingFootModel(supportFootIds, preImpact, comTask=comTask, 
                        centroidalTask=centroidalTask, swingFootTask=swingFootTask, forceTask=forceTask)
                    ]
        # Updating the current foot position for next step
        if not self.TRACK_CENTROIDAL:
            self.comRef += [stepLength * comPercentage, 0., 0.]
        for p in feetPos0:
            p += [stepLength, 0., 0.] 
        return footSwingModel 

    def createSwingFootModel(self, supportFootIds, preImpactTask=False, comTask=None, centroidalTask=None, swingFootTask=None, forceTask=None):
        # Creating a 3D multi-contact model, and then including the supporting feet
        nu = self.actuation.nu
        contactModel = crocoddyl.ContactModelMultiple(self.state, nu)
        # Creating the cost model for a contact phase
        costModel = crocoddyl.CostModelSum(self.state, nu)
        if isinstance(comTask, np.ndarray):
            self.add_com_position_tracking_cost(costModel, comTask)
        if swingFootTask is not None:
            if preImpactTask:
                self.add_swing_feet_impact_costs(costModel, swingFootTask)
            else:
                self.add_swing_feet_tracking_costs(costModel, swingFootTask)
        if isinstance(centroidalTask, np.ndarray):
            self.add_centroidal_momentum_tracking_cost(costModel, centroidalTask)
        self.add_support_contact_costs(contactModel, costModel, supportFootIds, forceTask)
        self.add_stat_ctrl_reg_costs(costModel, preImpactTask)
        # Creating the action model for the KKT dynamics with simpletic Euler integration scheme
        dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(self.state, self.actuation, contactModel,
                                                                                          costModel, 0., True)
        model = crocoddyl.IntegratedActionModelEuler(dmodel, self.dt)
        return model

    def get_solution_trajectories(self, solver):
        xs, us, K = solver.xs, solver.us, solver.K
        N = len(xs) 
        rmodel, rdata = self.rmodel, self.rdata
        jointPos_sol = np.empty((N, rmodel.nq))
        jointVel_sol = np.empty((N, rmodel.nv))
        jointTorques_sol = np.empty((N-1, rmodel.nv-6))
        centroidal_sol = np.empty((N, 9))
        gains = np.empty((N-1, K[0].shape[0], K[0].shape[1]))
        for time_idx in range (N):
            q, v = xs[time_idx][:rmodel.nq], xs[time_idx][rmodel.nq::] 
            pinocchio.framesForwardKinematics(rmodel, rdata, q)
            pinocchio.computeCentroidalMomentum(rmodel, rdata, q, v)
            centroidal_sol[time_idx, :3] = pinocchio.centerOfMass(rmodel, rdata, q, v)
            centroidal_sol[time_idx, 3:9] = np.array(rdata.hg)
            jointPos_sol[time_idx, :] = q
            jointVel_sol[time_idx, :] = v
            if time_idx < N-1:
                jointTorques_sol[time_idx, :] = us[time_idx]
                gains[time_idx, :,:] = K[time_idx]
        sol = {'centroidal':centroidal_sol, 'jointPos':jointPos_sol, 
               'jointVel':jointVel_sol, 'jointTorques':jointTorques_sol,
               'gains':gains}        
        return sol    

    def get_contact_positions_and_forces_solution(self, solver):
        contact_names = self.ee_frame_names
        contact_forces = np.zeros((len(solver.xs[:-1]), 3*len(contact_names)))
        contact_positions = np.zeros((len(solver.xs[:-1]), 3*len(contact_names)))
        for i, d in enumerate(solver.problem.runningDatas):
            pinocchio.framesForwardKinematics(
                self.rmodel, self.rdata, solver.xs[i][:self.rmodel.nq])
            m = solver.problem.runningModels[i]
            for k, c_key in enumerate(contact_names):
                c_id = self.rmodel.getFrameId(c_key)
                omf = self.rdata.oMf[c_id]
                contact_positions[i, 3*k:3*k+3] = np.resize(omf.translation, 3)
                try:
                    c_data = d.differential.multibody.contacts.contacts[c_key+'_contact']
                    contact_forces[i, 3*k:3*k+3] = np.resize(c_data.jMf.actInv(c_data.f).linear, 3)
                except:
                    pass
        return contact_positions, contact_forces

    def interpolate_whole_body_solution(self, solution):
        nq = self.rmodel.nq
        x, tau = solution['centroidal'], solution['jointTorques']
        q, qdot = solution['jointPos'], solution['jointVel']
        gains = solution['gains']
        N_inner = int(self.dt/self.dt_ctrl)
        N_outer_u  = tau.shape[0]
        N_outer_x  = x.shape[0]
        tau_interpol = np.empty((int((N_outer_u-1)*N_inner), tau.shape[1]))
        gains_interpol = np.empty((int((N_outer_u-1)*N_inner), gains.shape[1], gains.shape[2]))
        q_interpol = np.empty((int((N_outer_x-1)*N_inner), q.shape[1]))
        qdot_interpol = np.empty((int((N_outer_x-1)*N_inner), qdot.shape[1]))
        x_interpol = np.empty((int((N_outer_x-1)*N_inner), x.shape[1]))
        for i in range(N_outer_u-1):
            dtau = (tau[i+1] - tau[i])/N_inner
            #TODO find more elegant way to interpolate DDP gains 
            dgains = (gains[i+1]-gains[i])/N_inner
            for j in range(N_inner):
                k = i*N_inner + j
                tau_interpol[k] = tau[i] + j*dtau
                gains_interpol[k] = gains[i,:,:]+j*dgains
        for i in range(N_outer_x-1):
            dx = (x[i+1] - x[i])/N_inner
            dqdot = (qdot[i+1] - qdot[i])/N_inner
            for j in range(N_inner):
                k = i*N_inner + j
                x_interpol[k] = x[i] + j*dx
                if j == 0:
                    q_interpol[k] = q[i]
                else:
                    q_interpol[k] = pinocchio.interpolate(self.rmodel, q_interpol[k-1], q[i+1], self.dt_ctrl)
                qdot_interpol[k] = qdot[i] + j*dqdot
        interpol_sol =  {'centroidal':x_interpol, 'jointPos':q_interpol, 
                  'jointVel':qdot_interpol, 'jointTorques':tau_interpol,
                                                'gains':gains_interpol}               
        return interpol_sol

    # save solution in dat files for real robot experiments
    def save_solution_dat(self, solution):
        dt_ctrl = self.dt_ctrl
        q, qdot, tau = solution['jointPos'], solution['jointVel'], solution['jointTorques']
        time_x = np.arange(0, np.round(q.shape[0]*dt_ctrl, 2), dt_ctrl)
        time_u = np.arange(0, np.round(tau.shape[0]*dt_ctrl, 2), dt_ctrl)
        q_dat = np.column_stack((np.array([time_x, q[:, 0], q[:, 1], q[:, 2],q[:, 3],q[:, 4],q[:, 5],q[:, 6], q[:, 7],q[:, 8], q[:, 9],q[:, 10],q[:, 11]])))
        qdot_dat = np.column_stack((np.array([time_x, qdot[:, 0], qdot[:, 1], qdot[:, 2], qdot[:, 3], qdot[:, 4],qdot[:, 5],qdot[:, 6], qdot[:, 7],qdot[:, 8], qdot[:, 9],qdot[:, 10],q[:, 11]])))
        tau_dat = np.column_stack((np.array([time_u, tau[:, 0], tau[:, 1], tau[:, 2],tau[:, 3], tau[:, 4], tau[:, 5],tau[:, 6], tau[:, 7],tau[:, 8], tau[:, 9],tau[:, 10],tau[:, 11]])))
        np.savetxt('quadruped_positions.dat', q_dat, fmt=['%.8e','%.8e', '%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e'])
        np.savetxt('quadruped_velocities.dat', qdot_dat, fmt=['%.8e','%.8e', '%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e'])
        np.savetxt('quadruped_feedforward_torque.dat', tau_dat, fmt=['%.8e','%.8e', '%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e','%.8e'])

def plotSolution(solver, bounds=False, figIndex=1, figTitle="", show=True):
    import matplotlib.pyplot as plt
    xs, us = [], []
    if bounds:
        us_lb, us_ub = [], []
        xs_lb, xs_ub = [], []
    if isinstance(solver, list):
        rmodel = solver[0].problem.runningModels[0].state.pinocchio
        for s in solver:
            xs.extend(s.xs[:-1])
            us.extend(s.us)
            if bounds:
                models = s.problem.runningModels.tolist() + [s.problem.terminalModel]
                for m in models:
                    us_lb += [m.u_lb]
                    us_ub += [m.u_ub]
                    xs_lb += [m.state.lb]
                    xs_ub += [m.state.ub]
    else:
        rmodel = solver.problem.runningModels[0].state.pinocchio
        xs, us = solver.xs, solver.us
        if bounds:
            models = solver.problem.runningModels.tolist() + [solver.problem.terminalModel]
            for m in models:
                us_lb += [m.u_lb]
                us_ub += [m.u_ub]
                xs_lb += [m.state.lb]
                xs_ub += [m.state.ub]

    # Getting the state and control trajectories
    nx, nq, nu = xs[0].shape[0], rmodel.nq, us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    if bounds:
        U_LB = [0.] * nu
        U_UB = [0.] * nu
        X_LB = [0.] * nx
        X_UB = [0.] * nx
    for i in range(nx):
        X[i] = [np.asscalar(x[i]) for x in xs]
        if bounds:
            X_LB[i] = [np.asscalar(x[i]) for x in xs_lb]
            X_UB[i] = [np.asscalar(x[i]) for x in xs_ub]
    for i in range(nu):
        U[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else 0 for u in us]
        if bounds:
            U_LB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_lb]
            U_UB[i] = [np.asscalar(u[i]) if u.shape[0] != 0 else np.nan for u in us_ub]

    # Plotting the joint positions, velocities and torques
    plt.figure(figIndex)
    plt.suptitle(figTitle)
    legJointNames = ['HAA', 'HFE', 'KFE']
    # LF foot
    plt.subplot(4, 3, 1)
    plt.title('joint position [rad]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(7, 10))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(7, 10))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(7, 10))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(4, 3, 2)
    plt.title('joint velocity [rad/s]')
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 6, nq + 9))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 9))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 6, nq + 9))]
    plt.ylabel('LF')
    plt.legend()
    plt.subplot(4, 3, 3)
    plt.title('joint torque [Nm]')
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(0, 3))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(0, 3))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(0, 3))]
    plt.ylabel('LF')
    plt.legend()

    # LH foot
    plt.subplot(4, 3, 4)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(10, 13))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(10, 13))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(10, 13))]
    plt.ylabel('LH')
    plt.legend()
    plt.subplot(4, 3, 5)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 9, nq + 12))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 9, nq + 12))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 9, nq + 12))]
    plt.ylabel('LH')
    plt.legend()
    plt.subplot(4, 3, 6)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(3, 6))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(3, 6))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(3, 6))]
    plt.ylabel('LH')
    plt.legend()

    # RF foot
    plt.subplot(4, 3, 7)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(13, 16))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(13, 16))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(13, 16))]
    plt.ylabel('RF')
    plt.legend()
    plt.subplot(4, 3, 8)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 12, nq + 15))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 12, nq + 15))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 12, nq + 15))]
    plt.ylabel('RF')
    plt.legend()
    plt.subplot(4, 3, 9)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(6, 9))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(6, 9))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(6, 9))]
    plt.ylabel('RF')
    plt.legend()

    # RH foot
    plt.subplot(4, 3, 10)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(16, 19))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(16, 19))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(16, 19))]
    plt.ylabel('RH')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(4, 3, 11)
    [plt.plot(X[k], label=legJointNames[i]) for i, k in enumerate(range(nq + 15, nq + 18))]
    if bounds:
        [plt.plot(X_LB[k], '--r') for i, k in enumerate(range(nq + 15, nq + 18))]
        [plt.plot(X_UB[k], '--r') for i, k in enumerate(range(nq + 15, nq + 18))]
    plt.ylabel('RH')
    plt.xlabel('knots')
    plt.legend()
    plt.subplot(4, 3, 12)
    [plt.plot(U[k], label=legJointNames[i]) for i, k in enumerate(range(9, 12))]
    if bounds:
        [plt.plot(U_LB[k], '--r') for i, k in enumerate(range(9, 12))]
        [plt.plot(U_UB[k], '--r') for i, k in enumerate(range(9, 12))]
    plt.ylabel('RH')
    plt.legend()
    plt.xlabel('knots')

    plt.figure(figIndex + 1)
    plt.suptitle(figTitle)
    rdata = rmodel.createData()
    Cx = []
    Cy = []
    for x in xs:
        q = x[:nq]
        c = pinocchio.centerOfMass(rmodel, rdata, q)
        Cx.append(np.asscalar(c[0]))
        Cy.append(np.asscalar(c[1]))
    plt.plot(Cx, Cy)
    plt.title('CoM position')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid(True)
    if show:
        plt.show()

