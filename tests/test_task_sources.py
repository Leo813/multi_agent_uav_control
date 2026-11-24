"""Static preservation checks; these never instantiate a simulator task."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from uav_control.tasks import TASKS
from uav_control.tasks._compat import quat_apply_inverse
from uav_control.tasks.telemetry import write_legacy_csv


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "uav_control/tasks"


class WithoutTelemetry(ast.NodeTransformer):
    """Exclude only the documented optional CSV reset block and new gates/config."""
    def visit_AnnAssign(self, node):
        if isinstance(node.target, ast.Name) and node.target.id == "legacy_telemetry_dir":
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        condition = ast.unparse(node.test)
        if condition in ("torch.any(env_ids == 0)",
                         "self.cfg.legacy_telemetry_dir is not None and torch.any(env_ids == 0)"):
            return None
        if condition == "self.cfg.legacy_telemetry_dir is not None":
            return node.body
        return self.generic_visit(node)


def class_fingerprints(source):
    tree = WithoutTelemetry().visit(ast.parse(source, feature_version=(3, 10)))
    # Python 3.12 adds empty type_params fields even with the 3.10 grammar.
    return {node.name: hashlib.sha256(ast.dump(node).replace(", type_params=[]", "").encode()).hexdigest()
            for node in tree.body if isinstance(node, ast.ClassDef)}


@pytest.mark.parametrize("task", ["SARL", "MARL"])
def test_preserved_task_classes_and_configuration(task):
    source = (TASK_DIR / f"quadcopter_{task}.py").read_text(encoding="utf-8")
    provenance = json.loads((TASK_DIR / "provenance.json").read_text())
    assert class_fingerprints(source) == provenance["original_class_fingerprints"][task]
    assert 'CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")' in source
    assert "from isaaclab_assets import CRAZYFLIE_CFG" in source
    assert "from ._compat import quat_apply_inverse" in source
    assert "SPDX-License-Identifier: BSD-3-Clause" in source
    spec = TASKS[task]
    assert spec.id == f"UAV-Control-Quadcopter-{task}-v0"
    assert spec.entry_point.endswith(f":QuadcopterEnv{task}")
    assert spec.config_entry_point.endswith(f":QuadcopterEnvCfg{task}")
    assert not list((ROOT / "uav_control").rglob("*.usd*"))


@pytest.mark.parametrize("shape", [(1,), (8,), (2, 3)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quaternion_rotation_matches_unit_rotation_matrix(shape, dtype):
    generator = torch.Generator().manual_seed(91)
    q = torch.randn(*shape, 4, generator=generator, dtype=dtype)
    q = q / q.norm(dim=-1, keepdim=True)
    v = torch.randn(*shape, 3, generator=generator, dtype=dtype)
    w, x, y, z = q.unbind(-1)
    # Standard active rotation matrix; inverse is its transpose.
    matrix = torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                          2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                          2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*shape, 3, 3)
    expected = (matrix.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(quat_apply_inverse(q, v), expected)
    # Official 2.1.0 formula is equivalent for unit q, but differs otherwise.
    for scale in (1.0, 1.7):
        scaled = q * scale
        xyz, qw = scaled[..., 1:], scaled[..., :1]
        official = v*(2*qw.square()-1) - 2*qw*torch.cross(xyz, v, dim=-1) + 2*xyz*(xyz*v).sum(-1, keepdim=True)
        difference = 2*(1-scaled.square().sum(-1, keepdim=True))*v
        torch.testing.assert_close(quat_apply_inverse(scaled, v), official+difference)


def test_telemetry_safe_opt_in_and_ragged_columns(tmp_path):
    write_legacy_csv(None, "unused.csv", {"a": [1]})
    assert not list(tmp_path.iterdir())
    write_legacy_csv(tmp_path, "Total thrust:00001.csv", {"force": [2], "time": [0, 1]})
    assert (tmp_path / "Total thrust_00001.csv").read_text().splitlines() == ["force,time", "2,0", ",1"]
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    with pytest.warns(RuntimeWarning, match="optional telemetry"):
        write_legacy_csv(blocker, "data.csv", {"a": []})


def test_base_registration_and_launcher_imports_are_simulator_free():
    code = ("import sys; import uav_control, uav_control.tasks, uav_control.launcher; "
            "assert not any(n.split('.')[0] in ('isaaclab','isaacsim','omni','gymnasium','torch') "
            "for n in sys.modules)")
    subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, check=True)
