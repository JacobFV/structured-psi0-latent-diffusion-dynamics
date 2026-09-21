import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3

class BaseHelpers(unittest.TestCase):
    def helper(self, name):
        path = ROOT / 'scripts' / f'{name}.py'
        self.assertTrue(path.is_file(), f'{name} helper has not been implemented')
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def snapshot(self):
        return dict(memory_total_bytes=128*GIB, memory_available_bytes=40*GIB,
                    disk_total_bytes=1000*GIB, disk_free_bytes=200*GIB,
                    idle_cpu_cores=6.0)

    def policy(self):
        return json.loads((ROOT/'config/resources.json').read_text())

class HelperTests(BaseHelpers):
    def test_host_half_free_not_total(self):
        p = self.helper('preflight').calculate_budget(self.snapshot(), 'host', self.policy())
        self.assertEqual(p['memory_limit_bytes'], 20*GIB)
        self.assertEqual(p['cpu_quota_cores'], 3.0)
        self.assertLessEqual(p['new_disk_limit_bytes'], 100*GIB)

    def test_gpu_is_not_authorized(self):
        p = self.helper('preflight').calculate_budget(self.snapshot(), 'host', self.policy())
        self.assertFalse(p['heavy_work_authorized'])
        self.assertEqual(p['gpu_admission'], 'disabled_pending_enforcement')

    def test_peer_is_not_automatically_authorized(self):
        p = self.helper('preflight').calculate_budget(self.snapshot(), 'peer', self.policy())
        self.assertFalse(p['heavy_work_authorized'])
        self.assertEqual(p['memory_limit_bytes'], math.floor(min(0.8*40*GIB, 40*GIB-max(8*GIB,0.1*128*GIB))))

    def test_reserve_can_reduce_budget_to_zero(self):
        s = self.snapshot(); s['memory_available_bytes'] = 4*GIB
        p = self.helper('preflight').calculate_budget(s, 'host', self.policy())
        self.assertEqual(p['memory_limit_bytes'], 0)

    def test_disk_reserve_is_preserved(self):
        s = self.snapshot(); s['disk_free_bytes'] = 50*GIB
        p = self.helper('preflight').calculate_budget(s, 'host', self.policy())
        self.assertEqual(p['new_disk_limit_bytes'], 0)

    def test_fractional_cpu_not_rounded_up(self):
        s = self.snapshot(); s['idle_cpu_cores'] = 0.6
        p = self.helper('preflight').calculate_budget(s, 'host', self.policy())
        self.assertAlmostEqual(p['cpu_quota_cores'], 0.3)

    def test_host_fraction_cannot_exceed_half(self):
        mod = self.helper('preflight')
        for key in ['free_cpu_fraction','free_memory_fraction','free_disk_fraction']:
            with self.subTest(key=key):
                policy = self.policy(); policy['host'][key] = 0.51
                with self.assertRaises(ValueError): mod.calculate_budget(self.snapshot(),'host',policy)

    def test_bad_snapshot_numbers_rejected(self):
        mod = self.helper('preflight')
        for key in self.snapshot():
            for bad in [-1, float('nan'), float('inf'), True]:
                with self.subTest(key=key, bad=bad):
                    s = self.snapshot(); s[key] = bad
                    with self.assertRaises(ValueError): mod.calculate_budget(s,'host',self.policy())

    def test_inconsistent_availability_rejected(self):
        mod = self.helper('preflight')
        s=self.snapshot(); s['memory_available_bytes']=129*GIB
        with self.assertRaises(ValueError): mod.calculate_budget(s,'host',self.policy())

    def test_invalid_role_rejected(self):
        with self.assertRaises(ValueError):
            self.helper('preflight').calculate_budget(self.snapshot(),'all',self.policy())

    def test_meminfo_units(self):
        result=self.helper('preflight').parse_meminfo('MemTotal: 1024 kB\nMemAvailable: 512 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB\n')
        self.assertEqual(result['MemTotal'],1024*1024)
        self.assertEqual(result['MemAvailable'],512*1024)

    def test_missing_memavailable_is_not_guessed(self):
        with self.assertRaises(ValueError): self.helper('preflight').parse_meminfo('MemTotal: 1024 kB\nMemFree: 100 kB\n')

    def test_iowait_not_counted_as_idle(self):
        before={'cpu0':[0,0,0,0,0,0,0,0]}
        after={'cpu0':[10,0,0,40,50,0,0,0]}
        idle=self.helper('preflight').idle_cores_between(before,after,{0})
        self.assertAlmostEqual(idle,.4)

    def test_disallowed_cpu_not_counted(self):
        before={'cpu0':[0]*8, 'cpu1':[0]*8}
        after={'cpu0':[10,0,0,90,0,0,0,0],'cpu1':[0,0,0,100,0,0,0,0]}
        idle=self.helper('preflight').idle_cores_between(before,after,{0})
        self.assertAlmostEqual(idle,.9)

    def test_counter_reset_rejected(self):
        with self.assertRaises(ValueError):
            self.helper('preflight').idle_cores_between({'cpu0':[0,0,0,100,0,0,0,0]}, {'cpu0':[0]*8}, {0})

    def test_output_refuses_overwrite(self):
        mod=self.helper('preflight')
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'out.json'; p.write_text('keep')
            with self.assertRaises(FileExistsError): mod.write_json_exclusive(p,{'new':True})
            self.assertEqual(p.read_text(),'keep')

    def test_help_does_not_inspect_machine(self):
        self.helper('preflight')
        p=subprocess.run([sys.executable,str(ROOT/'scripts/preflight.py'),'--help'],capture_output=True,text=True,timeout=5)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('read-only',p.stdout.lower())

class IntegrityTests(BaseHelpers):
    def fixture_root(self, tmp):
        root=Path(tmp); (root/'a.txt').write_text('hello')
        digest=hashlib.sha256(b'hello').hexdigest()
        (root/'SHA256SUMS').write_text(f'{digest}  a.txt\n')
        return root

    def test_manifest_success(self):
        mod=self.helper('check_package')
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(mod.verify_manifest(self.fixture_root(tmp)),[])

    def test_manifest_detects_tamper(self):
        mod=self.helper('check_package')
        with tempfile.TemporaryDirectory() as tmp:
            root=self.fixture_root(tmp); (root/'a.txt').write_text('changed')
            self.assertTrue(mod.verify_manifest(root))

    def test_manifest_rejects_parent_traversal(self):
        mod=self.helper('check_package')
        with tempfile.TemporaryDirectory() as tmp:
            root=self.fixture_root(tmp)
            (root/'SHA256SUMS').write_text('0'*64+'  ../outside.txt\n')
            self.assertTrue(any('unsafe' in s.lower() for s in mod.verify_manifest(root)))

    def test_manifest_rejects_symlink(self):
        mod=self.helper('check_package')
        with tempfile.TemporaryDirectory() as tmp:
            root=self.fixture_root(tmp); (root/'b.txt').symlink_to(root/'a.txt')
            digest=hashlib.sha256(b'hello').hexdigest()
            (root/'SHA256SUMS').write_text(f'{digest}  b.txt\n')
            self.assertTrue(any('symlink' in s.lower() for s in mod.verify_manifest(root)))

    def test_duplicate_json_keys_rejected(self):
        mod=self.helper('check_package')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'a.json').write_text('{"x":1,"x":2}')
            self.assertTrue(mod.verify_json(root))

    def test_valid_json_accepted(self):
        mod=self.helper('check_package')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'a.json').write_text('{"x":1}')
            self.assertEqual(mod.verify_json(root),[])

if __name__ == '__main__':
    unittest.main()
