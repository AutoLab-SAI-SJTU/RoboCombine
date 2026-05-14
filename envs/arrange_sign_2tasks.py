from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
from ._GLOBAL_CONFIGS import asset_path

class arrange_sign_2tasks(Base_Task):

    def setup_demo(self, **kwargs):
        super()._init_task_env_(**kwargs)
        self.subtask_completed = [False, False]
        self.gripper_check_mode = kwargs.get('gripper_check_mode', 'full')

    def load_actors(self):
        import glob as _glob2
        _sort_objects = ['057_toycar', '073_rubikscube', '086_woodenblock', '107_soap']
        self.sort_modelname = np.random.choice(_sort_objects)
        _sp = str(asset_path('objects', self.sort_modelname))
        _jf = _glob2.glob(os.path.join(_sp, 'model_data*.json'))
        _sids = []
        for _f in _jf:
            try:
                _sids.append(int(os.path.basename(_f).replace('model_data', '').replace('.json', '')))
            except ValueError:
                pass
        if not _sids:
            _sids = [0]
        self.sort_model_id = np.random.choice(_sids)
        sort_pos = rand_pose(xlim=[0.15, 0.25], ylim=[-0.2, 0.0], qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, 3.14, 0])
        self.sort_object = create_actor(scene=self, pose=sort_pos, modelname=self.sort_modelname, convex=True, model_id=self.sort_model_id)
        self.sort_object.set_mass(0.05)
        self.sort_target_modelname = np.random.choice([x for x in _sort_objects if x != self.sort_modelname])
        _sp2 = str(asset_path('objects', self.sort_target_modelname))
        _jf2 = _glob2.glob(os.path.join(_sp2, 'model_data*.json'))
        _sids2 = []
        for _f2 in _jf2:
            try:
                _sids2.append(int(os.path.basename(_f2).replace('model_data', '').replace('.json', '')))
            except ValueError:
                pass
        if not _sids2:
            _sids2 = [0]
        self.sort_target_model_id = np.random.choice(_sids2)
        sort_tgt_pos = rand_pose(xlim=[0.15, 0.25], ylim=[-0.2, 0.0], qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, 3.14, 0])
        while np.sqrt((sort_tgt_pos.p[0] - sort_pos.p[0]) ** 2 + (sort_tgt_pos.p[1] - sort_pos.p[1]) ** 2) < 0.1:
            sort_tgt_pos = rand_pose(xlim=[0.15, 0.25], ylim=[-0.2, 0.0], qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=True, rotate_lim=[0, 3.14, 0])
        self.sort_target_obj = create_actor(scene=self, pose=sort_tgt_pos, modelname=self.sort_target_modelname, convex=True, model_id=self.sort_target_model_id)
        self.sort_target_obj.set_mass(0.05)
        self.add_prohibit_area(self.sort_object, padding=0.05)
        self.add_prohibit_area(self.sort_target_obj, padding=0.1)
        qrcode_pose = rand_pose(xlim=[-0.25, -0.15], ylim=[-0.2, 0.0], qpos=[0, 0, 0.707, 0.707], rotate_rand=True, rotate_lim=[0, 0.7, 0])
        self.qrcode_model_id = np.random.choice([0, 1, 2, 3], 1)[0]
        self.qrcode = create_actor(self, pose=qrcode_pose, modelname='070_paymentsign', convex=True, model_id=self.qrcode_model_id)
        self.add_prohibit_area(self.qrcode, padding=0.12)
        self.qrcode_target_pose = [-0.2, -0.15, 0.74 + self.table_z_bias, 1, 0, 0, 0]

    def play_once(self):
        print('[Subtask 1] Sort objects')
        arm_tag_sort = ArmTag('right')
        self.move(self.grasp_actor(self.sort_object, arm_tag=arm_tag_sort, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag_sort, z=0.1, move_axis='arm'))
        sort_target_p = self.sort_target_obj.get_pose().p.tolist()
        sort_target_p[0] += 0.13
        self.move(self.place_actor(self.sort_object, arm_tag=arm_tag_sort, target_pose=sort_target_p))
        print('[Subtask 2] Rotate QR code')
        arm_tag_qrcode = ArmTag('left')
        self.move(self.grasp_actor(self.qrcode, arm_tag=arm_tag_qrcode, pre_grasp_dis=0.05))
        self.move(self.move_by_displacement(arm_tag=arm_tag_qrcode, z=0.07))
        self.move(self.place_actor(self.qrcode, arm_tag=arm_tag_qrcode, target_pose=self.qrcode_target_pose, pre_dis=0.07, dis=0.01, constrain='align'))
        self.info['info'] = {'{A}': f'{self.sort_modelname}/base{self.sort_model_id}', '{B}': f'{self.sort_target_modelname}/base{self.sort_target_model_id}', '{a}': str(arm_tag_sort), '{C}': f'070_paymentsign/base{self.qrcode_model_id}', '{b}': str(arm_tag_qrcode)}
        return self.info

    def check_success(self):
        if not self.subtask_completed[0]:
            sort_obj_p = self.sort_object.get_pose().p
            sort_tgt_p = self.sort_target_obj.get_pose().p
            dist = np.sqrt(np.sum((sort_obj_p[:2] - sort_tgt_p[:2]) ** 2))
            if dist < 0.2 and dist > 0.08 and (sort_obj_p[0] > sort_tgt_p[0]) and (abs(sort_obj_p[1] - sort_tgt_p[1]) < 0.05):
                print('  Subtask 1 completed: Object sorted')
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
