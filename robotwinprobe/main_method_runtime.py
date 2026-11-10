"""Process-local relocation of project paths embedded in RoboTwin YAML assets.

No vendor code, shared YAML, URDF, collision geometry or model data is edited.
Install before task/planner imports so safe_load and CuRobo's CLoader path agree.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import bootstrap as B

OLD_ROOTS = ('/home/user1/workspace/chence/WAMattack',
             '/shared/user65/workspace/chence/WAMattack')
_installed = None


def relocate(value, root=None, changes=None):
    """Rewrite only complete known absolute-project prefixes in string values."""
    root = Path(B.ROOT if root is None else root)
    if isinstance(value, str):
        for old in OLD_ROOTS:
            if value == old or value.startswith(old+'/'):
                rewritten = str(root)+value[len(old):]
                if rewritten != value and changes is not None:
                    changes[value] = rewritten
                return rewritten
        return value
    if isinstance(value, dict):
        return {k:relocate(v,root,changes) for k,v in value.items()}
    if isinstance(value, list):
        return [relocate(v,root,changes) for v in value]
    if isinstance(value, tuple):
        return tuple(relocate(v,root,changes) for v in value)
    return value


def install_path_normalization(out=None):
    """Intercept YAML parsing in this process, before RoboTwin task imports."""
    global _installed
    import yaml
    if _installed is not None:
        if _installed['root'] != str(B.ROOT):
            raise RuntimeError('runtime project root changed after path hook installation')
        return _installed
    original = yaml.load
    changes = {}
    report = dict(root=str(B.ROOT),known_prefixes=list(OLD_ROOTS),
                  scope='process-local yaml.load result normalization; assets unmodified',
                  normalized_paths=changes)
    path = None if out is None else Path(out)/'runtime_paths.json'
    def save():
        if path is not None:
            path.parent.mkdir(parents=True,exist_ok=True)
            temporary=path.with_suffix('.tmp.json')
            temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(path)
    def normalized_load(stream, Loader):
        previous=len(changes)
        result=relocate(original(stream,Loader=Loader),B.ROOT,changes)
        if len(changes)!=previous:save()
        return result
    yaml.load=normalized_load
    _installed=report
    save()
    return report


def audit_required_assets():
    """CPU-only proof of normalized planner files and relative URDF meshes."""
    import yaml
    base=B.ROBOTWIN/'assets/embodiments/aloha-agilex'
    checks=[]
    urdfs=set()
    for filename in ['config.yml','curobo_left.yml','curobo_right.yml']:
        source=base/filename
        with source.open() as f:data=yaml.safe_load(f)
        nodes = data['robot_cfg']['kinematics'] if 'robot_cfg' in data else data
        for key in ['urdf_path','srdf_path','collision_spheres']:
            value=nodes.get(key)
            if not isinstance(value,str):continue
            path=Path(value)
            if not path.is_absolute():path=(base/path).resolve()
            assert path.is_file(),('required asset missing',filename,key,str(path))
            checks.append(dict(source=str(source),key=key,path=str(path),exists=True))
            if key=='urdf_path':urdfs.add(path)
    meshes=[]
    for urdf in sorted(urdfs):
        for node in ET.parse(urdf).iter('mesh'):
            filename=node.attrib.get('filename')
            if not filename:continue
            if filename.startswith('package://'):
                raise RuntimeError('unresolved package URI in required URDF: '+filename)
            path=Path(relocate(filename))
            if not path.is_absolute():path=(urdf.parent/path).resolve()
            assert path.is_file(),('required URDF mesh missing',str(urdf),filename,str(path))
            meshes.append(str(path))
    return dict(passed=True,root=str(B.ROOT),required_paths=checks,
                urdfs=[str(p) for p in sorted(urdfs)],mesh_references=len(meshes),
                unique_mesh_files=sorted(set(meshes)),gpu_context_created=False,
                note='Only path strings were relocated; parsed numeric/collision/kinematic values are unchanged')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out')
    args=ap.parse_args();install_path_normalization(args.out)
    result=audit_required_assets()
    if args.out:
        p=Path(args.out);p.mkdir(parents=True,exist_ok=True)
        (p/'asset_path_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
