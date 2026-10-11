from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np

class kitchen_switch_2tasks(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        self.subtask_completed = [False, False]
        self.gripper_check_mode = kwargs.get('gripper_check_mode', 'full')

    def load_actors(self):
        container_pose = rand_pose(xlim=[0.2, 0.28], ylim=[0.0, 0.1], rotate_rand=False, qpos=[0.5, 0.5, 0.5, 0.5])
        id_list = {'002_bowl': [1, 2, 3, 5], '021_cup': [1, 2, 3, 4, 5, 6, 7]}
        self.actor_name = np.random.choice(['002_bowl', '021_cup'])
        self.container_id = np.random.choice(id_list[self.actor_name])
        self.container = create_actor(self, pose=container_pose, modelname=self.actor_name, model_id=self.container_id, convex=True)
        self.plate_id = 0
        plate_pose = rand_pose(xlim=[0.15, 0.2], ylim=[-0.2, -0.15], rotate_rand=False, qpos=[0.5, 0.5, 0.5, 0.5])
        self.plate = create_actor(self, pose=plate_pose, modelname='003_plate', scale=[0.025, 0.025, 0.025], is_static=True, convex=True)
        self.add_prohibit_area(self.container, padding=0.1)
        self.add_prohibit_area(self.plate, padding=0.1)
        self.switch_model_id = np.random.randint(0, 8)
        self.switch = rand_create_sapien_urdf_obj(scene=self, modelname='056_switch', modelid=self.switch_model_id, xlim=[-0.25, -0.05], ylim=[0.0, 0.1], zlim=[0.81, 0.84], rotate_rand=True, rotate_lim=[0, 0, np.pi / 4], qpos=[0.704141, 0, 0, 0.71006], fix_root_link=True)
        self.prohibited_area.append([-0.4, -0.2, 0.4, 0.2])

    def play_once(self):
        print('[Subtask 1] Place container on plate')
        arm_tag_container = ArmTag('right')
        self.move(self.grasp_actor(self.container, arm_tag=arm_tag_container, contact_point_id=0, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag_container, z=0.1, move_axis='arm'))
        plate_fp = self.plate.get_functional_point(0)
        if plate_fp is None:
            plate_fp = self.plate.get_pose()
        self.move(self.place_actor(self.container, target_pose=plate_fp, arm_tag=arm_tag_container, functional_point_id=0, pre_dis=0.12, dis=0.03))
        self.move(self.move_by_displacement(arm_tag_container, z=0.08, move_axis='arm'))
        print('[Subtask 2] Turn switch')
        arm_tag_switch = ArmTag('left')
        self.move(self.close_gripper(arm_tag=arm_tag_switch, pos=0))
        self.move(self.grasp_actor(self.switch, arm_tag=arm_tag_switch, pre_grasp_dis=0.04))
        self.move(self.open_gripper(arm_tag=arm_tag_switch))
        self.move(self.move_by_displacement(arm_tag_switch, z=0.08, move_axis='arm'))
        self.info['info'] = {'{A}': '003_plate/base0', '{B}': f'{self.actor_name}/base{self.container_id}', '{a}': str(arm_tag_container), '{C}': f'056_switch/base{self.switch_model_id}', '{b}': str(arm_tag_switch)}
        return self.info

    def check_success(self):
        if not self.subtask_completed[0]:
            container_pose = self.container.get_pose().p
            target_pose = self.plate.get_pose().p
            eps = np.array([0.05, 0.05, 0.03])
            if np.all(abs(container_pose[:3] - target_pose) < eps):
                print('  Subtask 1 completed: Container placed on plate')
                self.subtask_completed[0] = True
        if not self.subtask_completed[1]:
            limit = self.switch.get_qlimits()[0]
            if self.switch.get_qpos()[0] >= limit[1] - 0.05:
                print('  Subtask 2 completed: Switch turned')
                self.subtask_completed[1] = True
        left_gripper_val = self.robot.left_gripper_val
        right_gripper_val = self.robot.right_gripper_val
        gripper_check_mode = getattr(self, 'gripper_check_mode', 'full')
        if gripper_check_mode == 'none':
            left_open = True
            right_open = True
            gripper_open = True
        elif gripper_check_mode == 'relaxed':
            left_open = left_gripper_val > 0.6
            right_open = right_gripper_val > 0.05
            gripper_open = left_open and right_open
        else:
            left_open = self.robot.is_left_gripper_open()
            right_open = self.robot.is_right_gripper_open()
            gripper_open = left_open and right_open
        current_step = self.take_action_cnt
        success = all(self.subtask_completed) and gripper_open
        if not hasattr(self, '_last_logged_step'):
            self._last_logged_step = -1
        if current_step != self._last_logged_step:
            self._last_logged_step = current_step
            mode_indicator = {'full': '', 'relaxed': '[R]', 'none': '[N]'}[gripper_check_mode]
            print(f'[Step {current_step:4d}]{mode_indicator} S1:{(chr(10003) if self.subtask_completed[0] else chr(10007))} S2:{(chr(10003) if self.subtask_completed[1] else chr(10007))} L:{left_gripper_val:.3f}({(chr(10003) if left_open else chr(10007))}) R:{right_gripper_val:.3f}({(chr(10003) if right_open else chr(10007))}) -> ', end='')
            if success:
                print('SUCCESS!')
            else:
                reasons = []
                if not self.subtask_completed[0]:
                    reasons.append('S1')
                if not self.subtask_completed[1]:
                    reasons.append('S2')
                if all(self.subtask_completed) and gripper_check_mode != 'none':
                    if not left_open:
                        reasons.append(f"L<{(0.6 if gripper_check_mode == 'relaxed' else 0.8)}")
                    if not right_open:
                        reasons.append(f"R<{(0.05 if gripper_check_mode == 'relaxed' else 0.8)}")
                print(f"FAIL: {','.join(reasons)}")
        return success
