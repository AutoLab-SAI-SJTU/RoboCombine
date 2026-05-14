from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np

class checkout_counter_2tasks(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        self.subtask_completed = [False, False]
        self.gripper_check_mode = kwargs.get('gripper_check_mode', 'full')

    def load_actors(self):
        ori_quat = [[0.707, 0.707, 0, 0], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, -0.5, -0.5], [0.5, 0.5, -0.5, -0.5], [0.5, -0.5, 0.5, -0.5]]
        self.phone_id = np.random.choice([0, 1, 2, 4], 1)[0]
        phone_pose = rand_pose(xlim=[0.15, 0.25], ylim=[-0.2, 0.0], qpos=ori_quat[self.phone_id], rotate_rand=True, rotate_lim=[0, 0.7, 0])
        self.phone = create_actor(scene=self, pose=phone_pose, modelname='077_phone', convex=True, model_id=self.phone_id)
        self.phone.set_mass(0.01)
        stand_pose = rand_pose(xlim=[0.05, 0.15], ylim=[0, 0.2], qpos=[0.707, 0.707, 0, 0], rotate_rand=False)
        while np.sqrt(np.sum((phone_pose.p[:2] - stand_pose.p[:2]) ** 2)) < 0.15:
            stand_pose = rand_pose(xlim=[0.05, 0.15], ylim=[0, 0.2], qpos=[0.707, 0.707, 0, 0], rotate_rand=False)
        self.stand_id = np.random.choice([1, 2], 1)[0]
        self.stand = create_actor(scene=self, pose=stand_pose, modelname='078_phonestand', convex=True, model_id=self.stand_id, is_static=True)
        self.add_prohibit_area(self.phone, padding=0.15)
        self.add_prohibit_area(self.stand, padding=0.15)
        qrcode_pose = rand_pose(xlim=[-0.25, -0.15], ylim=[-0.2, 0.0], qpos=[0, 0, 0.707, 0.707], rotate_rand=True, rotate_lim=[0, 0.7, 0])
        self.qrcode_model_id = np.random.choice([0, 1, 2, 3], 1)[0]
        self.qrcode = create_actor(self, pose=qrcode_pose, modelname='070_paymentsign', convex=True, model_id=self.qrcode_model_id)
        self.add_prohibit_area(self.qrcode, padding=0.12)
        self.qrcode_target_pose = [-0.2, -0.15, 0.74 + self.table_z_bias, 1, 0, 0, 0]

    def play_once(self):
        print('[Subtask 1] Place phone on stand')
        arm_tag_phone = ArmTag('right')
        self.move(self.grasp_actor(self.phone, arm_tag=arm_tag_phone, pre_grasp_dis=0.08))
        stand_func_pose = self.stand.get_functional_point(0)
        self.move(self.place_actor(self.phone, arm_tag=arm_tag_phone, target_pose=stand_func_pose, functional_point_id=0, dis=0, constrain='align'))
        print('[Subtask 2] Rotate QR code')
        arm_tag_qrcode = ArmTag('left')
        self.move(self.grasp_actor(self.qrcode, arm_tag=arm_tag_qrcode, pre_grasp_dis=0.05))
        self.move(self.move_by_displacement(arm_tag=arm_tag_qrcode, z=0.07))
        self.move(self.place_actor(self.qrcode, arm_tag=arm_tag_qrcode, target_pose=self.qrcode_target_pose, pre_dis=0.07, dis=0.01, constrain='align'))
        self.info['info'] = {'{A}': f'077_phone/base{self.phone_id}', '{B}': f'078_phonestand/base{self.stand_id}', '{a}': str(arm_tag_phone), '{C}': f'070_paymentsign/base{self.qrcode_model_id}', '{b}': str(arm_tag_qrcode)}
        return self.info

    def check_success(self):
        if not self.subtask_completed[0]:
            phone_fp = np.array(self.phone.get_functional_point(0))
            stand_fp = np.array(self.stand.get_functional_point(0))
            eps = np.array([0.045, 0.04, 0.04])
            if np.all(np.abs(phone_fp - stand_fp)[:3] < eps):
                print('  Subtask 1 completed: Phone placed on stand')
                self.subtask_completed[0] = True
        if not self.subtask_completed[1]:
            qrcode_quat = self.qrcode.get_pose().q
            qrcode_pos = self.qrcode.get_pose().p
            target_quat = [0.707, 0.707, 0, 0]
            if qrcode_quat[0] < 0:
                qrcode_quat = qrcode_quat * -1
            eps = 0.05
            rot_ok = np.all(np.abs(qrcode_quat - target_quat) < eps)
            z_ok = qrcode_pos[2] < 0.75 + self.table_z_bias
            if rot_ok and z_ok:
                print('  Subtask 2 completed: QR code rotated')
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
