from edge_environment import *

# 常量
c1: float = 2
c2: float = 2
r1: float = 0.5
r2: float = 0.5
w: float = 1.1
LAN: float = 12.5  # 100Mbps局域网
WAN: float = 1.25  # 10Mbps广域网


class ParticleSwarm:
	'''
	粒子群核心算法模块
	'''

	def __init__(self, env: EdgeEnvironment, iter=200, pNum=30):
		'''
		初始化粒子群
		:param env: 绑定的边缘计算环境
		:param iter: 粒子群算法迭代次数
		:param pNum: 粒子数
		'''
		self.env = env
		self.iter_num = iter
		self.particle_num = pNum
		self.particle_swarm, self.velocity, self.best_particle = self.init_pso()
		self.fitness_value = np.full(
			shape=(self.particle_num,),
			fill_value=np.inf,  # 适应度值初始化为无穷大
			dtype=np.float
		)
		self.global_best_particle = -1

	def init_pso(self):
		'''
		根据边缘计算环境初始化粒子群和速度
		:return:由粒子数、工作流数、任务数3个维度组成的三维数组
		'''
		particle_swarm = np.random.randint(
			low=0,
			high=self.env.device_num,
			size=self.env.workflow_num * self.env.task_num * self.particle_num
		).reshape((
			self.particle_num,
			self.env.workflow_num,
			self.env.task_num
		))

		velocity = np.random.randint(
			low=-self.env.device_num,
			high=self.env.device_num,
			size=self.env.workflow_num * self.env.task_num * self.particle_num
		).reshape((
			self.particle_num,
			self.env.workflow_num,
			self.env.task_num
		))

		best_particle = np.copy(particle_swarm)

		return particle_swarm, velocity, best_particle

	def update_velocity(self):
		'''
		更新粒子的速度
		:return:
		'''
		for i in range(self.particle_num):
			rand1 = np.random.random(self.env.workflow_num * self.env.task_num).reshape((
				self.env.workflow_num,
				self.env.task_num
			))
			rand2 = np.random.random(self.env.workflow_num * self.env.task_num).reshape((
				self.env.workflow_num,
				self.env.task_num
			))  # 产生两个二维随机数组
			part = w * self.velocity[i] + c1 * rand1 * (self.best_particle[i] - self.particle_swarm[i]) \
			       + c2 * rand2 * (self.best_particle[self.global_best_particle] - self.particle_swarm[i])
			self.velocity[i] = part.astype(np.int)
			for j in range(self.env.workflow_num):
				for k in range(self.env.task_num):
					if self.velocity[i, j, k] > self.env.device_num:
						self.velocity[i, j, k] = 0  # 速度超过了上限则置零

	def update_position(self):
		'''
		更新粒子的位置
		:return:
		'''
		self.particle_swarm = np.mod(self.particle_swarm + self.velocity, self.env.device_num)

	def fitness(self):
		'''
		适应度函数
		:return:
		'''
		time = self.get_time()
		energy = self.get_energy()
		bool_deadline = time < self.env.workflow_task_deadline
		fit = np.zeros_like(self.particle_swarm, dtype=np.int64)

		for i in range(self.particle_num):
			for j in range(self.env.workflow_num):
				for k in range(self.env.task_num):
					if bool_deadline[i, j, k]:
						fit[i, j, k] = energy[i, j, k]  # 没超deadline
					else:
						fit[i, j, k] = 10 * energy[i, j, k] * time[i, j, k] / self.env.workflow_task_deadline[j, k]

		return fit.sum(axis=(1, 2))  # 对workflow和task两个维度求和，保留particle维度的一维数组

	def update_best(self):
		'''
		更新每个粒子的最优值
		:return: None
		'''
		for i in range(self.particle_num):
			if self.fitness()[i] < self.fitness_value[i]:
				for j in range(self.env.workflow_num):
					for k in range(self.env.task_num):
						self.best_particle[i, j, k] = self.particle_swarm[i, j, k]

		self.fitness_value = np.minimum(self.fitness_value, self.fitness())

	def update_global_best(self):
		'''
		更新粒子的全局最优值
		:return: None
		'''
		self.global_best_particle = self.fitness_value.argmin()

	def get_time(self):
		'''
		获取当前分配模式下每个粒子每个工作流的每个任务的时间
		:return:三维数组
		'''
		workload = np.broadcast_to(
			array=self.env.workflow_task_workload,
			shape=(
				self.particle_num,
				self.env.workflow_num,
				self.env.task_num
			)
		)  # 每个工作流的每个任务的负载扩展为三维数组

		procRate = np.zeros_like(workload)
		for i in range(self.particle_num):
			for j in range(self.env.workflow_num):
				for k in range(self.env.task_num):
					# 根据下标寻找服务器从而获取处理速率
					procRate[i, j, k] = self.env.device_procRate[self.particle_swarm[i, j, k]]

		proc_time = workload / procRate  # 计算执行时间

		# 接受时间和发送时间初始化if
		recv_time = np.zeros_like(self.particle_swarm)
		recv_time.flags.writeable = True
		send_time = np.zeros_like(self.particle_swarm)
		send_time.flags.writeable = True

		for i in range(self.particle_num):
			for j in range(self.env.workflow_num):
				for k in range(self.env.task_num):
					# 获取设备类型，任务前驱、后继等信息
					device = self.particle_swarm[i, j, k]
					type = self.env.device_type[device]
					prior = self.env.task_prior[k]
					next = self.env.task_next[k]
					inputSize = self.env.workflow_task_inputSize[j, k]
					outputSize = self.env.workflow_task_outputSize[j, k]

					# 计算接收时间
					if len(prior) == 0:
						# 这是第一个任务
						if type == 'EndDevice':
							# 第一个任务未卸载（仍在终端执行）
							recv_time[i, j, k] = 0
						else:
							if type == 'CloudServer':
								# 端云之间为广域网
								recv_time[i, j, k] = inputSize / WAN
							else:
								recv_time[i, j, k] = inputSize / LAN
					else:
						if len(prior) == 1:
							# 只有一个前驱，不需要判断去最大值
							if device == self.particle_swarm[i, j, prior[0]]:
								# 先判断前面的任务是否与当前任务在同一设备上
								recv_time[i, j, k] = 0
							else:
								if type == 'CloudServer' or self.particle_swarm[i, j, prior[0]] == 'CloudServer':
									# 只要有一个设备是云就是广域网传输
									recv_time[i, j, k] = inputSize / WAN
								else:
									recv_time[i, j, k] = inputSize / LAN
						else:
							max_recv_time = []
							for x in prior:
								# 当前任务的输入数据大小经由其前驱任务的输出数据大小分别计算
								prior_outputSize = self.env.workflow_task_outputSize[j, x]
								if device == self.particle_swarm[i, j, x]:
									# 先判断前面的任务是否与当前任务在同一设备上
									max_recv_time.append(0)
								else:
									if type == 'CloudServer' or self.particle_swarm[i, j, x] == 'CloudServer':
										# 只要有一个设备是云就是广域网传输
										max_recv_time.append(prior_outputSize / WAN)
									else:
										max_recv_time.append(prior_outputSize / LAN)
							recv_time[i, j, k] = max(max_recv_time)  # 根据关键路径原理，取最大值

					# 计算发送时间
					if len(next) == 0:
						# 最后一个任务执行完成
						if type == 'EndDevice':
							# 最后一个任务已经在终端中
							send_time[i, j, k] = 0
						else:
							if type == 'CloudServer':
								# 端云之间为广域网
								send_time[i, j, k] = outputSize / WAN
							else:
								send_time[i, j, k] = outputSize / LAN
					else:
						if len(next) == 1:
							# 只有一个后继，不需要去判断最大值
							if device == self.particle_swarm[i, j, next[0]]:
								recv_time[i, j, k] = 0
							else:
								if type == 'CloudServer' or self.particle_swarm[i, j, next[0]] == 'CloudServer':
									recv_time[i, j, k] = outputSize / WAN
								else:
									recv_time[i, j, k] = outputSize / LAN
						else:
							max_send_time = []
							for y in next:
								next_inputSize = self.env.workflow_task_inputSize[j, y]
								if device == self.particle_swarm[i, j, y]:
									max_send_time.append(0)
								else:
									if type == 'CloudServer' or self.particle_swarm[i, j, y] == 'CloudServer':
										max_send_time.append(next_inputSize / WAN)
									else:
										max_send_time.append(next_inputSize / LAN)
							send_time[i, j, k] = max(max_send_time)

		return recv_time + proc_time + send_time

	def get_energy(self):
		'''
		获取终端设备的能耗
		:return: 三维数组
		'''
		device_energy = np.zeros_like(self.particle_swarm)
		for i in range(self.particle_num):
			for j in range(self.env.workflow_num):
				for k in range(self.env.task_num):
					if self.env.device_type[self.particle_swarm[i, j, k]] == 'EndDevice':
						# 仅考虑终端设备能耗
						# 获取设备，前驱后继、输入输出数据信息
						device = self.particle_swarm[i, j, k]
						prior = self.env.task_prior[k]
						next = self.env.task_next[k]
						inputSize = self.env.workflow_task_inputSize[j, k]
						outputSize = self.env.workflow_task_outputSize[j, k]
						workload = self.env.workflow_task_workload[j, k]
						procRate = self.env.device_procRate[device]

						# 计算执行能耗=时间×功率
						proc_energy = workload / procRate * self.env.end_device['procPower']

						# 计算接收能耗
						recv_energy = 0
						if len(prior) == 0:
							# 这是第一个任务，而且没有卸载
							recv_energy = 0
						else:
							if len(prior) == 1:
								# 只有一个前驱，需要计算接收时间
								if device == self.particle_swarm[i, j, prior[0]]:
									# 先判断前面的任务是否与当前任务在同一设备上
									recv_energy = 0
								else:
									if self.particle_swarm[i, j, prior[0]] == 'CloudServer':
										recv_energy = inputSize / WAN * self.env.end_device['recvPower']
									else:
										recv_energy = inputSize / LAN * self.env.end_device['recvPower']
							else:
								max_recv_time = []
								for x in prior:
									# 当前任务的输入数据大小经由其前驱任务的输出数据大小分别计算
									prior_outputSize = self.env.workflow_task_outputSize[j, x]
									if device == self.particle_swarm[i, j, x]:
										# 先判断前面的任务是否与当前任务在同一设备上
										max_recv_time.append(0)
									else:
										if self.particle_swarm[i, j, x] == 'CloudServer':
											max_recv_time.append(prior_outputSize / WAN)
										else:
											max_recv_time.append(prior_outputSize / LAN)
								recv_energy = max(max_recv_time) * self.env.end_device['recvPower']

						# 计算发送能耗
						send_energy = 0
						if len(next) == 0:
							send_energy = 0
						else:
							if len(next) == 1:
								if device == self.particle_swarm[i, j, next[0]]:
									send_energy = 0
								else:
									if self.particle_swarm[i, j, next[0]] == 'CloudServer':
										send_energy = outputSize / WAN * self.env.end_device['sendPower']
									else:
										send_energy = outputSize / LAN * self.env.end_device['sendPower']
							else:
								max_send_time = []
								for y in next:
									next_inputSize = self.env.workflow_task_inputSize[j, y]
									if device == self.particle_swarm[i, j, y]:
										max_send_time.append(0)
									else:
										if self.particle_swarm[i, j, y] == 'CloudServer':
											max_send_time.append(next_inputSize / WAN)
										else:
											max_send_time.append(next_inputSize / LAN)
								send_energy = max(max_send_time) * self.env.end_device['sendPower']

						# 计算总能耗
						device_energy[i, j, k] = proc_energy + recv_energy + send_energy

		return device_energy

	def start_pso(self):
		'''
		pso算法开始迭代
		:return:
		'''
		for i in range(self.iter_num):
			print('开始第%d次迭代' % (i + 1))
			self.update_best()
			self.update_global_best()
			self.update_velocity()
			self.update_position()


ee = EdgeEnvironment('./YOLO1.json', 100)
pp = ParticleSwarm(ee)
pp.start_pso()
print(pp.best_particle[pp.global_best_particle])
