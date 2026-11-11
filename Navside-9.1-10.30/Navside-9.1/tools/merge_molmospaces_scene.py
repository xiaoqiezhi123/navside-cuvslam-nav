#!/usr/bin/env python3
import argparse
import copy
import re
import xml.etree.ElementTree as ET
from pathlib import Path


REF_ATTRS = {
    "class",
    "material",
    "mesh",
    "texture",
    "hfield",
    "site",
    "body",
    "geom",
    "joint",
    "tendon",
    "actuator",
    "objname",
    "target",
    "child",
    "parent",
}

SRU_DEMO_PREFIXES = (
    "streetlight_",
    "obstacle_",
)

SRU_DEMO_NAMES = {
    "road_ground",
    "target_marker_body",
}


def parse_xml(path: Path) -> ET.ElementTree:
    text = path.read_text(encoding="utf-8")
    # Some exported MJCF files are accepted by MuJoCo but are not strict XML
    # because adjacent attributes can miss whitespace, e.g. a="1"b="2".
    text = re.sub(r'(?<=")(?=[A-Za-z_][A-Za-z0-9_.:-]*=)', " ", text)
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    return ET.ElementTree(ET.fromstring(text, parser=parser))


def child(root: ET.Element, tag: str) -> ET.Element | None:
    return root.find(tag)


def ensure_child(root: ET.Element, tag: str) -> ET.Element:
    existing = child(root, tag)
    if existing is not None:
        return existing
    return ET.SubElement(root, tag)


def compiler_dir(root: ET.Element, xml_path: Path, attr: str) -> Path:
    compiler = child(root, "compiler")
    raw = compiler.get(attr) if compiler is not None else None
    if not raw and attr == "texturedir":
        raw = compiler.get("meshdir") if compiler is not None else None
    if not raw:
        return xml_path.parent
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (xml_path.parent / path).resolve()


def first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def absolutize_asset_paths(root: ET.Element, xml_path: Path) -> None:
    compiler = ensure_child(root, "compiler")
    mesh_dir = compiler_dir(root, xml_path, "meshdir")
    texture_dir = compiler_dir(root, xml_path, "texturedir")
    asset_dir = compiler_dir(root, xml_path, "assetdir")
    compiler.set("meshdir", str(mesh_dir))

    for elem in root.iter():
        if "file" not in elem.attrib:
            continue
        file_path = Path(elem.get("file", "")).expanduser()
        if file_path.is_absolute():
            elem.set("file", str(file_path))
            continue

        base_dir = xml_path.parent
        if elem.tag == "mesh":
            base_dir = mesh_dir
            candidates = [base_dir / file_path, xml_path.parent / file_path]
        elif elem.tag == "texture":
            candidates = [xml_path.parent / file_path, texture_dir / file_path]
        elif elem.tag == "hfield":
            base_dir = asset_dir
            candidates = [base_dir / file_path, xml_path.parent / file_path]
        else:
            candidates = [base_dir / file_path]
        elem.set("file", str(first_existing(candidates).resolve()))


def prefix_scene_names(root: ET.Element, prefix: str) -> None:
    renamed: dict[str, str] = {}

    def new_name(name: str) -> str:
        safe = name.replace("/", "_").replace(" ", "_")
        return f"{prefix}{safe}"

    for elem in root.iter():
        name = elem.get("name")
        if name:
            renamed[name] = new_name(name)
            elem.set("name", renamed[name])
        class_name = elem.get("class")
        if elem.tag == "default" and class_name:
            renamed[class_name] = new_name(class_name)
            elem.set("class", renamed[class_name])

    for elem in root.iter():
        for attr in REF_ATTRS:
            value = elem.get(attr)
            if value in renamed:
                elem.set(attr, renamed[value])


def should_drop_sru_demo_child(elem: ET.Element) -> bool:
    name = elem.get("name", "")
    if name in SRU_DEMO_NAMES:
        return True
    return any(name.startswith(prefix) for prefix in SRU_DEMO_PREFIXES)


def clear_sru_demo_scene(root: ET.Element) -> None:
    worldbody = child(root, "worldbody")
    if worldbody is None:
        return
    for elem in list(worldbody):
        if should_drop_sru_demo_child(elem):
            worldbody.remove(elem)


def merge_section(dst_root: ET.Element, src_root: ET.Element, tag: str) -> int:
    src = child(src_root, tag)
    if src is None:
        return 0
    dst = ensure_child(dst_root, tag)
    count = 0
    for elem in list(src):
        dst.append(copy.deepcopy(elem))
        count += 1
    return count


def set_robot_spawn(root: ET.Element, spawn_xyz: str | None, spawn_yaw: str | None) -> None:
    if spawn_xyz is None and spawn_yaw is None:
        return

    worldbody = child(root, "worldbody")
    if worldbody is None:
        raise ValueError("SRU XML has no <worldbody>")

    base = None
    for elem in worldbody.iter("body"):
        if elem.get("name") == "base_link":
            base = elem
            break
    if base is None:
        raise ValueError("SRU XML has no body named base_link")

    if spawn_xyz is not None:
        parts = [float(part.strip()) for part in spawn_xyz.split(",")]
        if len(parts) != 3:
            raise ValueError("--robot-spawn-xyz must be x,y,z")
        base.set("pos", "{:.6g} {:.6g} {:.6g}".format(*parts))

        keyframe = child(root, "keyframe")
        if keyframe is not None:
            key = keyframe.find("key")
            if key is not None and key.get("qpos"):
                qpos = [float(v) for v in key.get("qpos", "").split()]
                if len(qpos) >= 3:
                    qpos[0:3] = parts
                    key.set("qpos", " ".join("{:.6g}".format(v) for v in qpos))

    if spawn_yaw is not None:
        yaw = float(spawn_yaw)
        base.set("euler", "0 0 {:.6g}".format(yaw))


def merge_scene(
    sru_xml: Path,
    scene_xml: Path,
    output_xml: Path,
    prefix: str,
    clear_demo_scene: bool,
    robot_spawn_xyz: str | None,
    robot_spawn_yaw: str | None,
) -> tuple[int, int, int]:
    sru_tree = parse_xml(sru_xml)
    scene_tree = parse_xml(scene_xml)
    sru_root = sru_tree.getroot()
    scene_root = scene_tree.getroot()

    absolutize_asset_paths(sru_root, sru_xml)
    absolutize_asset_paths(scene_root, scene_xml)
    prefix_scene_names(scene_root, prefix)

    if clear_demo_scene:
        clear_sru_demo_scene(sru_root)
    set_robot_spawn(sru_root, robot_spawn_xyz, robot_spawn_yaw)

    default_count = merge_section(sru_root, scene_root, "default")
    asset_count = merge_section(sru_root, scene_root, "asset")
    world_count = merge_section(sru_root, scene_root, "worldbody")

    sru_root.set("model", "phybot_molmospaces_scene")
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(sru_tree, space="  ")
    sru_tree.write(output_xml, encoding="utf-8", xml_declaration=False)
    return default_count, asset_count, world_count


def resolve_from_script(path_text: str, script_dir: Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (script_dir / path).resolve()


def absolute_without_following_symlink(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path.absolute()
    return (Path.cwd() / path).absolute()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a MolmoSpaces MuJoCo scene into the SRU NavSide robot XML."
    )
    parser.add_argument(
        "--sru-xml",
        default="../asset/robot/phybot_mini_mark2/xml/phybot_real.xml",
        help="SRU robot MuJoCo XML.",
    )
    parser.add_argument("--scene-xml", required=True, help="MolmoSpaces scene XML.")
    parser.add_argument(
        "--output-xml",
        default="../asset/merged/phybot_molmospaces_scene.xml",
        help="Merged output XML.",
    )
    parser.add_argument("--prefix", default="mls_", help="Prefix for scene names.")
    parser.add_argument(
        "--keep-sru-demo-scene",
        action="store_true",
        help="Keep existing SRU demo street obstacles instead of replacing them.",
    )
    parser.add_argument(
        "--robot-spawn-xyz",
        default=None,
        help='Optional SRU base spawn as "x,y,z".',
    )
    parser.add_argument(
        "--robot-spawn-yaw",
        default=None,
        help="Optional SRU base yaw in radians.",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    sru_xml = resolve_from_script(args.sru_xml, script_dir)
    scene_xml = absolute_without_following_symlink(args.scene_xml)
    output_xml = resolve_from_script(args.output_xml, script_dir)

    default_count, asset_count, world_count = merge_scene(
        sru_xml=sru_xml,
        scene_xml=scene_xml,
        output_xml=output_xml,
        prefix=args.prefix,
        clear_demo_scene=not args.keep_sru_demo_scene,
        robot_spawn_xyz=args.robot_spawn_xyz,
        robot_spawn_yaw=args.robot_spawn_yaw,
    )
    print(f"sru_xml={sru_xml}")
    print(f"scene_xml={scene_xml}")
    print(f"output_xml={output_xml}")
    print(
        "merged "
        f"default_children={default_count} asset_children={asset_count} "
        f"worldbody_children={world_count}"
    )


if __name__ == "__main__":
    main()
