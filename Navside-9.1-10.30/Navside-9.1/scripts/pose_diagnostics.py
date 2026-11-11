#!/usr/bin/env python3
"""Passive pose diagnostics: prepare, watch, report, existing. No robot/network IO."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'logs/pose_diag'


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare(out=None):
    BASE.mkdir(parents=True, exist_ok=True)
    directory = Path(out).expanduser().resolve() if out else BASE/('run_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name in ('nav_pathaware.yaml', 'nav_deploy.yaml', 'task_nav.yaml', 'task_points_selected.json'):
        src = ROOT/'config'/name
        if src.is_file():
            shutil.copy2(src, directory/name)
            hashes[name] = hashlib.sha256(src.read_bytes()).hexdigest()
    write_json(directory/'manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        config_sha256=hashes, note='只准备诊断；尚未启动任何进程。'))
    activation = (f'export NAV_POSE_DIAG_DIR={shlex.quote(str(directory))}\n'
                  f'export NAV_POSE_DIAG_MODULE={shlex.quote(str(ROOT/"navside/pose_trace.py"))}\n')
    (directory/'enable.sh').write_text(activation)
    (BASE/'enable_latest.sh').write_text(activation)
    (BASE/'latest.txt').write_text(str(directory)+'\n')
    print(f'诊断目录：{directory}\n同一终端先执行：source logs/pose_diag/enable_latest.sh\n'
          '再按原命令启动调度器。观察：python3 scripts/pose_diagnostics.py watch --latest\n'
          '报告：python3 scripts/pose_diagnostics.py report --latest')
    return directory


def read_rows(directory):
    rows, malformed = [], 0
    for path in sorted(directory.glob('*.jsonl')):
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except (ValueError, TypeError):
                    malformed += 1
    rows.sort(key=lambda row: row.get('mono_ns', 0))
    return rows, malformed


def key(row):
    return (row.get('stream_id'), row.get('source_sequence'))


def delta(a, b):
    return (a-b)/1e9 if a and b else None


def distance(a, b):
    if a is None or b is None or not all(math.isfinite(x) for x in (*a, *b)):
        return None
    return math.dist(a, b)


def same_vio_frame(a, b):
    """Local coordinates reset even if localization failed (no new anchor_epoch)."""
    if a is None or b is None or a['logger_id'] != b['logger_id']:
        return False
    # Generations also distinguish a new object that reuses an old Python id.
    if a.get('tracker_generation') is not None or b.get('tracker_generation') is not None:
        return a.get('tracker_generation') == b.get('tracker_generation')
    # Older traces lack generations. Use object identity where it was recorded.
    return a.get('tracker_id') == b.get('tracker_id')


def statistics(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return {'count': 0, 'max': None, 'p95': None}
    return {'count': len(values), 'max': max(values), 'p95': values[math.ceil(.95*len(values))-1]}


def analyze(rows):
    tx = {key(r): r for r in rows if r['stage'] == 'tx' and r.get('stream_id')}
    vio = {(r['logger_id'], r['source_timestamp_ns']): r for r in rows if r['stage'] == 'vio'}
    boots = {r['logger_id']: r.get('boot_id') for r in rows if r['stage'] == 'trace_start'}
    starts = {(r['logger_id'], r['inference_id']): r for r in rows if r['stage'] == 'inference_start'}
    uses = {}
    for row in starts.values():
        uses.setdefault(key(row), []).append(row)
    rx = [r for r in rows if r['stage'] == 'rx']
    previous, samples, mismatches, raw_events = {}, [], [], []
    matched, received_keys = 0, set()
    for row in rx:
        sid = row.get('stream_id') or row['logger_id']
        prev = previous.get(sid)
        source = tx.get(key(row))
        raw = vio.get((sid, row.get('source_timestamp_ns')))
        prev_raw = vio.get((sid, prev.get('source_timestamp_ns'))) if prev else None
        same_epoch = prev is not None and row.get('anchor_epoch') == prev.get('anchor_epoch')
        clock_ok = bool(source and boots.get(source['logger_id']) and
                        boots.get(source['logger_id']) == boots.get(row['logger_id']))
        pose = row['wire_pose']
        item = dict(stream_id=sid, source_sequence=row.get('source_sequence'),
            pose_sequence=row.get('pose_sequence'), anchor_epoch=row.get('anchor_epoch'),
            receive_time_utc=stamp(row), receive_wall_ns=row.get('wall_ns'), source_timestamp_ns=row.get('source_timestamp_ns'),
            frame_received_ns=row.get('frame_received_ns'), send_ns=row.get('send_ns'),
            socket_received_ns=row.get('socket_received_ns'), received_ns=row.get('received_ns'),
            x=pose[0], y=pose[1], z_raw=pose[2], qw=pose[3], qx=pose[4], qy=pose[5], qz=pose[6],
            source_dt_s=delta(row.get('source_timestamp_ns'), prev.get('source_timestamp_ns')) if prev else None,
            send_dt_s=delta(row.get('send_ns'), prev.get('send_ns')) if prev else None,
            receive_dt_s=delta(row.get('socket_received_ns'), prev.get('socket_received_ns')) if prev else None,
            send_to_receive_s=delta(row.get('socket_received_ns'), row.get('send_ns')) if clock_ok else None,
            receive_to_cache_s=delta(row.get('received_ns'), row.get('socket_received_ns')),
            frame_to_send_s=delta(row.get('send_ns'), row.get('frame_received_ns')),
            horizontal_step_m=distance(pose[:2], prev['wire_pose'][:2]) if same_epoch else None,
            global_step_3d_m=distance(pose[:3], prev['wire_pose'][:3]) if same_epoch else None,
            tracker_generation=raw.get('tracker_generation') if raw else None,
            raw_vio_frame_continuous=same_vio_frame(raw, prev_raw),
            raw_vio_step_3d_m=distance(raw['local_position'], prev_raw['local_position']) if same_vio_frame(raw, prev_raw) and same_epoch else None,
            source_sequence_gap=(row['source_sequence']-prev['source_sequence']-1) if prev and row.get('source_sequence') else None,
            inference_use_count=len(uses.get(key(row), [])) if row.get('stream_id') else None, clock_comparable=clock_ok)
        sample_uses = uses.get(key(row), []) if row.get('stream_id') else []
        item['first_inference_age_s'] = delta(sample_uses[0]['mono_ns'], row.get('received_ns')) if sample_uses else None
        if source:
            matched += 1
            received_keys.add(key(row))
            if source['wire_pose'] != pose:
                mismatches.append(dict(kind='tx_rx_pose_mismatch', stream_id=sid, seq=row.get('source_sequence')))
        for use in sample_uses:
            # SRU intentionally overwrites Z=0.695 and casts coordinates to float32.
            if any(abs(a-b) > 1e-5*max(1, abs(a)) for a,b in zip(pose[:2]+pose[3:], use['state_position'][:2]+use['state_quaternion'])):
                mismatches.append(dict(kind='rx_sru_pose_mismatch', stream_id=sid, seq=row.get('source_sequence')))
        samples.append(item)
        previous[sid] = row
    prev_raw = {}
    for r in rows:
        if r['stage'] != 'vio':
            continue
        prev = prev_raw.get(r['logger_id'])
        step = distance(r['local_position'], prev['local_position']) if same_vio_frame(r, prev) else None
        raw_events.append(dict(time_utc=stamp(r), wall_ns=r.get('wall_ns'), mono_ns=r['mono_ns'], logger_id=r['logger_id'],
            source_timestamp_ns=r['source_timestamp_ns'], local_x=r['local_position'][0],
            tracker_id=r.get('tracker_id'), tracker_generation=r.get('tracker_generation'),
            local_frame_continuous=same_vio_frame(r, prev),
            track_cpu_s=r.get('track_cpu_s'), imu_wait_and_register_s=r.get('imu_wait_and_register_s'),
            stereo_timestamp_delta_s=delta(r.get('right_source_timestamp_ns'),r['source_timestamp_ns']),
            local_y=r['local_position'][1], local_z=r['local_position'][2], observations=r.get('observations'),
            source_dt_s=delta(r['source_timestamp_ns'], prev['source_timestamp_ns']) if prev else None,
            sdk_receive_dt_s=delta(r['frame_received_ns'], prev['frame_received_ns']) if prev else None,
            sdk_wait_s=delta(r['frame_received_ns'], r.get('frame_wait_started_ns')),
            preprocessing_and_imu_s=delta(r['track_started_ns'], r['frame_received_ns']),
            track_s=delta(r['track_done_ns'], r['track_started_ns']), local_step_m=step))
        prev_raw[r['logger_id']] = r
    ends = [r for r in rows if r['stage'] == 'inference_end']
    summary = dict(tx_count=len(tx), rx_count=len(rx), matched_tx_rx=matched,
        tx_without_rx=len(set(tx)-received_keys), rx_without_tx=len(rx)-matched,
        legacy_rx_count=sum(r.get('wire_version', 1)==1 for r in rx), mismatches=mismatches,
        metrics={k:statistics(s.get(k) for s in samples) for k in
            ('source_dt_s','send_dt_s','receive_dt_s','send_to_receive_s','receive_to_cache_s','frame_to_send_s',
             'horizontal_step_m','global_step_3d_m','raw_vio_step_3d_m','first_inference_age_s')},
        vio_metrics={k:statistics(s.get(k) for s in raw_events) for k in
                     ('source_dt_s','sdk_receive_dt_s','sdk_wait_s','preprocessing_and_imu_s','track_s','local_step_m')},
        inference_duration_s=statistics(r['duration_s'] for r in ends),
        inference_end_pose_age_s=statistics(delta(r['mono_ns'],r.get('received_ns')) for r in ends),
        visualize_duration_s=statistics(r['duration_s'] for r in rows if r['stage']=='visualize'),
        imu_source_dt_s=statistics(r.get('source_dt_s') for r in rows if r['stage']=='imu'),
        image_to_last_imu_s=statistics(r.get('image_to_last_imu_s') for r in rows if r['stage']=='imu_register'),
        imu_register_without_samples=sum(r.get('registered')==0 for r in rows if r['stage']=='imu_register'),
        source_sequence_anomalies=[s for s in samples if s['source_sequence_gap'] not in (None,0)],
        largest_steps=sorted([s for s in samples if s['horizontal_step_m'] is not None],
                             key=lambda s:s['horizontal_step_m'],reverse=True)[:10],
        trace_dropped_by_logger={sid:max(r.get('trace_dropped',0) for r in rows if r['logger_id']==sid)
                                 for sid in {r['logger_id'] for r in rows}},
        unclosed_loggers=sorted({r['logger_id'] for r in rows if r['stage']=='trace_start'}
                                - {r['logger_id'] for r in rows if r['stage']=='trace_end'}),
        path_events=[r for r in rows if r['stage']=='path_event'],
        active_configs=[r for r in rows if r['stage']=='config'])
    return summary, samples, raw_events


def stamp(row):
    value = row.get('wall_ns') or row.get('receive_wall_ns')
    return datetime.fromtimestamp(value/1e9, timezone.utc).strftime('%H:%M:%S.%f')[:-3] if value else '时刻未记录'


def number(value, unit='s'):
    if value is None or not math.isfinite(value):
        return '无数据'
    return f'{value*1000:.2f} ms' if unit == 'ms' else f'{value:.3f} {unit}'


def cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def interpret(rows, summary, samples):
    """Explain evidence only; never infer physical motion or change navigation."""
    rows = sorted(rows, key=lambda r: r.get('mono_ns', 0))
    boots = {r['logger_id']: r.get('boot_id') for r in rows if r['stage'] == 'trace_start'}
    controls = [r for r in rows if r['stage'] in ('control', 'anchor')]
    tx = {key(r): r for r in rows if r['stage'] == 'tx'}
    rx = [r for r in rows if r['stage'] == 'rx']
    vio = [r for r in rows if r['stage'] == 'vio']
    events = summary['path_events']

    def phase(row):
        previous = [r for r in controls if r['logger_id'] == row['logger_id']
                    and r['mono_ns'] <= row['mono_ns']]
        if not previous:
            return '启动/阶段未记录'
        last = previous[-1]
        return '恢复输出期间' if last.get('command') == 'resume' else '暂停/重定位期间'

    # Source controls and source TX share a clock; never compare different hosts' clocks.
    intervals, previous = [], {}
    for row, sample in zip(rx, samples):
        sid = row.get('stream_id') or row['logger_id']
        prev = previous.get((row['logger_id'], sid))
        previous[(row['logger_id'], sid)] = row
        if not prev:
            continue
        start, end = tx.get(key(prev)), tx.get(key(row))
        label = '阶段证据不足'
        if row.get('anchor_epoch') != prev.get('anchor_epoch'):
            label = '跨暂停/重定位'
        elif start and end:
            boundary = any(r['logger_id'] == start['logger_id'] and
                           start['mono_ns'] < r['mono_ns'] <= end['mono_ns'] and
                           (r['stage'] == 'anchor' or r.get('command') == 'pause') for r in controls)
            if boundary:
                label = '跨暂停/重定位'
            elif phase(start) == '恢复输出期间' and phase(end) == '恢复输出期间':
                label = '连续输出区间'
        intervals.append(dict(start=stamp(prev), end=stamp(row), kind=label,
                              receive_dt_s=sample['receive_dt_s'], send_dt_s=sample['send_dt_s']))

    waits = []
    for event in events:
        if event.get('path') != 'waiting_pose':
            continue
        future = [r for r in events if r['logger_id'] == event['logger_id'] and
                  r['mono_ns'] > event['mono_ns'] and r.get('path') in
                  ('pose_resumed', 'error', 'paused', 'stopped', 'ready', 'running')]
        outcome = future[0] if future else None
        if outcome and outcome.get('segment_id') != event.get('segment_id'):
            outcome = None
        before = [r for r in rx if r['logger_id'] == event['logger_id'] and
                  r['mono_ns'] <= event['mono_ns'] and r.get('pose_sequence') == event.get('pose_sequence')]
        last = before[-1] if before else None
        after = [r for r in rx if last and r['logger_id'] == event['logger_id'] and
                 r['mono_ns'] > event['mono_ns'] and r.get('stream_id') == last.get('stream_id') and
                 r.get('anchor_epoch') == last.get('anchor_epoch') and
                 (not outcome or r['mono_ns'] <= outcome['mono_ns'])]
        next_rx = after[0] if after else None
        configs = [r for r in summary['active_configs'] if r['logger_id'] == event['logger_id'] and r['mono_ns'] <= event['mono_ns']]
        item = dict(event=event, outcome=outcome, last_rx=last, next_rx=next_rx, slow_track=None,
                    max_age_s=configs[-1].get('state_max_age_s') if configs else None,
                    receive_gap_s=None, send_gap_s=None, next_delivery_s=None, upstream_gap=False)
        if last and next_rx:
            item['receive_gap_s'] = delta(next_rx.get('socket_received_ns'), last.get('socket_received_ns'))
            a, b = tx.get(key(last)), tx.get(key(next_rx))
            if a and b and boots.get(a['logger_id']) and boots.get(a['logger_id']) == boots.get(event['logger_id']):
                item['send_gap_s'] = delta(b.get('send_ns'), a.get('send_ns'))
                item['next_delivery_s'] = delta(next_rx.get('socket_received_ns'), b.get('send_ns'))
                candidates = [r for r in vio if r['logger_id'] == a['logger_id'] and
                              a['mono_ns'] <= r['track_started_ns'] < r['track_done_ns'] <= b['mono_ns']]
                if candidates:
                    slow = max(candidates, key=lambda r: r['track_done_ns']-r['track_started_ns'])
                    duration = delta(slow['track_done_ns'], slow['track_started_ns'])
                    item['slow_track'] = dict(time=stamp(slow), duration_s=duration)
                    gap, delivery = item['receive_gap_s'], item['next_delivery_s']
                    item['upstream_gap'] = bool(gap and gap > 0 and duration > gap/2 and
                                                delivery is not None and 0 <= delivery < gap/10)
        waits.append(item)

    tracker_checks = []
    for r in rows:
        if r['stage'] != 'main_tracker_created':
            continue
        following = next((v for v in vio if v['logger_id'] == r['logger_id'] and v['mono_ns'] > r['mono_ns']), None)
        if following and r.get('tracker_id') is not None and following.get('tracker_id') is not None:
            tracker_checks.append(dict(time=stamp(r), main=r['tracker_id'], worker=following['tracker_id']))
    imu = [dict(time=stamp(r), phase=phase(r), lag_s=r['image_to_last_imu_s']) for r in rows
           if r['stage'] == 'imu_register' and r.get('image_to_last_imu_s') is not None]
    received_keys = {key(r) for r in rx if r.get('stream_id')}
    generations = []
    for switched in (r for r in rows if r['stage'] == 'tracker_switched'):
        frames = [v for v in vio if v['logger_id'] == switched['logger_id'] and
                  v.get('tracker_generation') == switched.get('tracker_generation') and v['mono_ns'] >= switched['mono_ns']]
        generations.append(dict(time=stamp(switched), generation=switched.get('tracker_generation'),
                                tracker_id=switched.get('tracker_id'), frame_count=len(frames),
                                mismatch_count=sum(v.get('tracker_id') != switched.get('tracker_id') for v in frames)))
    return dict(waits=waits, intervals=intervals, tracker_checks=tracker_checks,
                tracker_generations=generations,
                imu_rejected_frames=sum(r['stage'] == 'imu_frame_rejected' for r in rows),
                largest_imu_lags=sorted(imu, key=lambda r:r['lag_s'], reverse=True)[:3],
                inference_count=sum(r['stage'] == 'inference_start' for r in rows),
                matched_inferences=sum(r['stage'] == 'inference_start' and key(r) in received_keys for r in rows),
                raw_vio_count=len(vio), anchors=sum(r['stage'] == 'anchor' for r in rows))


def related_task_evidence(rows):
    """Read only task logs proven to belong to the trace's session ID."""
    ids = {r.get('session_id') for r in rows if r.get('session_id') not in (None, '', 'none')}
    evidence = []
    if not ids:
        return evidence
    for path in sorted((ROOT/'logs/task_nav/astar').glob('run_*/session.json')):
        try:
            session = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        sid = session.get('session_id')
        if sid not in ids:
            continue
        item = dict(session_id=sid, session_file=str(path), failures=[], arrivals=[], localize_failed=False)
        failures = path.parent/'failures.jsonl'
        if failures.is_file():
            for line in failures.read_text().splitlines():
                try:
                    item['failures'].append(json.loads(line))
                except ValueError:
                    continue
            item['failures_file'] = str(failures)
        # Verify the NAV file contains this exact session before relating the paired SLAM log.
        match = re.fullmatch(r'run_(\d{8}_\d{6})_\d+', path.parent.name)
        if match:
            suffix = match[1]
            nav = ROOT/f'logs/task_nav/nav_sched_{suffix}.log'
            slam = ROOT/f'logs/task_nav/slam_sched_{suffix}.log'
            if nav.is_file():
                lines = [line for line in nav.read_text(errors='replace').splitlines()
                         if re.search(r'\bsession_id='+re.escape(sid)+r'(?:\s|$)', line)]
                if lines:
                    item['nav_file'] = str(nav)
                    item['arrivals'] = [line for line in lines if 'zero_reason=goal_reached' in line]
                    if slam.is_file():
                        item['slam_file'] = str(slam)
                        item['localize_failed'] = '[localize] FAILED' in slam.read_text(errors='replace')
        evidence.append(item)
    return evidence


def render_report(directory, rows, summary, samples, findings, related):
    events = summary['path_events']
    waits = findings['waits']
    manual = [r for r in events if r.get('path') == 'paused' and r.get('reason') == 'operator']
    errors = [r for r in events if r.get('path') == 'error']
    scheduler = [r for r in events if r.get('path') == 'stopped' and r.get('reason') == 'scheduler']
    configs = summary['active_configs']
    max_age = configs[-1].get('state_max_age_s') if configs else None
    recovery = configs[-1].get('pose_recovery', {}) if configs else {}
    dated = [r for r in rows if r.get('wall_ns')]
    span = ('无时间记录' if not dated else
            datetime.fromtimestamp(min(r['wall_ns'] for r in dated)/1e9, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')+
            ' ～ '+datetime.fromtimestamp(max(r['wall_ns'] for r in dated)/1e9, timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))
    full = bool(summary['tx_count'] and summary['rx_count'] and findings['inference_count'])
    lines = ['# 位姿诊断报告：本轮结论与排查步骤', '', f'**采集时间：{span}（UTC）。**',
             f'数据目录：`{directory}`', '',
             '本报告分析这个目录内已有的记录。`report --latest` 不会补采数据；先核对上面的采集时间是否就是刚才那轮。', '',
             '## 1. 先看结论', '']
    if not full:
        lines += ['**数据不足：本轮尚未覆盖完整的“发送 → 接收 → SRU 推理”链路，不能判断下游是否正常。**',
                  f'原始 VIO {findings["raw_vio_count"]} 帧，成功锚定 {findings["anchors"]} 次；'
                  f'发送 {summary["tx_count"]} 包，接收 {summary["rx_count"]} 包，推理开始 {findings["inference_count"]} 次。',
                  '若本来就用 `--no-sru`，缺少接收/推理记录符合该模式；若要检查完整链路，先确认定位成功、下游出现 `running`。', '']
    elif summary['mismatches']:
        lines += ['**本轮发现发送/接收或 SRU 输入坐标不一致，优先核对收发与状态组装。** 具体包编号见 `summary.json → mismatches`。', '']
    elif any(w['upstream_gap'] for w in waits):
        lines += ['**本轮至少一次自动等待位姿的证据指向上游 `track()` 调用耗时过长，造成一段时间没有新位姿。** '
                  '对应空窗内发送也停了，新包发出后很快被接收；优先查上游跟踪及其运行负载。', '']
    elif errors:
        lines += ['**本轮存在下游异常停车。先看下面的具体原因与时刻；目前不能仅凭停车名称判定根因。**', '']
    elif waits:
        lines += ['**本轮出现自动等待位姿，但现有证据不足以把原因归到某一个环节。** 下面按事件列出已采到的证据。', '']
    else:
        lines += ['**本轮记录中未发现自动等待位姿或下游 `error` 事件。** 这不等于 VIO 精度已经验证通过。', '']
    lines += ['| 要判断的问题 | 本轮结论 | 证据与限制 |', '|---|---|---|']
    if summary['mismatches']:
        link_status, link_detail = '发现数据不一致，需查收发/状态组装', f'{len(summary["mismatches"])} 条，具体包编号见 summary.json 的 mismatches'
    elif summary['matched_tx_rx']:
        link_status = '已匹配的发送/接收坐标一致'
        link_detail = (f'精确关联 {summary["matched_tx_rx"]} 包；'
                       f'发送→接收最大 {number(summary["metrics"]["send_to_receive_s"]["max"], "ms")}。'
                       + (f'与接收包关联的 {findings["matched_inferences"]} 次 SRU 输入 XY/姿态也一致。'
                          if findings['matched_inferences'] else '尚无可关联的 SRU 输入证据。'))
    else:
        link_status, link_detail = '数据不足', '没有可逐包匹配的发送/接收记录'
    lines.append(f'| 坐标是否在发送/接收时被改错？ | {link_status} | {link_detail} |')
    resumed = sum(w['outcome'] is not None and w['outcome'].get('path') == 'pose_resumed' for w in waits)
    lines += [f'| 是否触发自动等待？ | 记录到 {len(waits)} 次等待，{resumed} 次自动恢复 | 详见第 2 节；没有事件不能证明未发生未记录的故障 |',
              f'| 是否发生需要 localize 的下游异常停车？ | 记录到 {len(errors)} 条 error | 人工暂停另列，不能把所有 stopped 当异常 |',
              '| VIO 是否漂移？ | **目前不能确认** | 坐标数值一致只能说明传递一致；必须与现场静止或已知距离/朝向对照 |']
    planning = [f for x in related for f in x['failures']]
    lines += [f'| A* / 地图是否导致过暂停？ | '+
              (f'关联记录中有 {len(planning)} 次规划失败' if planning else
               ('关联记录未见规划失败' if related else '未找到同会话规划日志，不能判断'))+
              ' | 与位姿超时是两类问题，具体原因见第 2 节 |', '', '## 2. 这次为什么停车', '']
    for i, w in enumerate(waits, 1):
        e, outcome = w['event'], w['outcome']
        lines += [f'### 自动等待 {i}：{stamp(e)}，`{cell(e.get("reason", "未记录"))}`', '',
                  f'- 触发时位姿包龄 **{number(e.get("pose_age_s"))}**；该进程当时的超时门槛 **{number(w["max_age_s"])}**。'
                  '下游进入等待状态并保留当前目标/路径。']
        if w['last_rx'] and w['next_rx']:
            lines += [f'- 最后一个包：{stamp(w["last_rx"])}；下一个包：{stamp(w["next_rx"])}。'
                      f'接收间隔 **{number(w["receive_gap_s"])}**，发送间隔 **{number(w["send_gap_s"])}**；'
                      f'新包的发送→接收耗时 **{number(w["next_delivery_s"], "ms")}**。']
        if w['slow_track']:
            lines += [f'- 同一空窗内，原始 VIO 的一次 `track()` 耗时 **{number(w["slow_track"]["duration_s"])}**。'
                      '这是调用墙钟耗时，包含可能的 GPU 等待、同步和调度，尚不能细分到哪一项。']
        lines += ['- 判断：'+('主要空窗已经出现在上游发包之前，跟踪调用耗时覆盖了空窗的大部分；优先排查上游。'
                             if w['upstream_gap'] else '现有记录尚不能确认等待的根因，按第 5 节检查缺失或变慢的环节。')]
        if outcome and outcome.get('path') == 'pose_resumed':
            lines += [f'- 结果：**{stamp(outcome)} 自动恢复**，等待 {number(outcome.get("wait_s"))}，'
                      f'稳定样本 {outcome.get("stable_samples", "未记录")} 个。此事件没有要求人工 localize。']
        elif outcome:
            lines += [f'- 后续：{stamp(outcome)}，`{cell(outcome.get("path"))}` / `{cell(outcome.get("reason", "未记录"))}`；未确认自动恢复。']
        else:
            lines += ['- 后续：未采到同一路径的结束事件，无法确认恢复或最终停车原因。']
        lines.append('')
    if not waits:
        lines += ['本轮未记录 `waiting_pose`；若缺少 NAV 日志，则无法据此排除等待事件。', '']
    lines += ['### 其他停车与流程事件', '', '| 时间（UTC） | 类型 | 应如何理解 |', '|---|---|---|']
    for e in errors:
        lines.append(f'| {stamp(e)} | 下游异常 `{cell(e.get("reason", "未记录"))}` | '
                     f'偏移 {number(e.get("position_jump_m"), "m")}；判定阈值 {number(e.get("position_limit_m"), "m")}。'
                     '先核对原因与数据，按原流程 localize 后重试当前目标 |')
    for e in manual:
        lines.append(f'| {stamp(e)} | 人工暂停 `operator` | 按键触发；必须 localize，不能自动恢复。同一次停留可能多次按键 |')
    for item in related:
        for failure in item['failures']:
            lines.append(f'| {cell(failure.get("time_utc", "时刻未记录"))} | A* 规划失败 | {cell(failure.get("error", "未记录"))}。'
                         '这是规划可通行性问题，需在规划图核对实际重定位起点/目标位置 |')
        for arrival in item['arrivals']:
            dist = re.search(r'goal_dist=([\d.]+)', arrival)
            lines.append('| 关联日志未记录精确时刻 | 到达判定 `goal_reached` | 下游按目标到达条件发零速；'
                         + (f'记录距离 {dist[1]} m；' if dist else '')+'不是位姿异常停车 |')
        if item['localize_failed']:
            lines.append('| 关联日志未记录精确时刻 | 重定位失败 | SLAM 记录 `[localize] FAILED`；需查视觉匹配/初始猜测，不能归因于 A* 墙体厚度 |')
    lines += ['', f'另外有 {len(scheduler)} 条 `stopped / scheduler`：表示调度器发了停车命令，'
              '可出现在人工暂停、到点、重定位、规划失败等流程中，**不是额外的故障次数**。',
              '协议 `segment_id` 的尾号是路径下发版本；重规划同一个目标也会增加，不能直接当作任务点编号。', '',
              '## 3. 哪些大数值容易看错', '']
    for kind, title in [('连续输出区间', '排除已知暂停/重定位后，连续输出区间的最大接收间隔'),
                        ('跨暂停/重定位', '跨暂停/重定位的最大接收间隔'), ('阶段证据不足', '阶段不明的最大接收间隔')]:
        values = [x for x in findings['intervals'] if x['kind'] == kind and x['receive_dt_s'] is not None]
        if values:
            worst = max(values, key=lambda x:x['receive_dt_s'])
            lines.append(f'- **{title}：{number(worst["receive_dt_s"])}**（{worst["start"]} → {worst["end"]}）。'+
                         ('这段时间包含主动停止发位姿/定位等待，不能当成一个 UDP 包传输了这么久。' if kind == '跨暂停/重定位' else
                          '它是两次收包之间的间隔，不是单包传输延迟。'))
    if summary['largest_steps']:
        s = summary['largest_steps'][0]
        lines += [f'- **最大相邻出口水平位移：{number(s["horizontal_step_m"], "m")}**，发生于 {stamp(s)}，'
                  f'源包序号 {s["source_sequence"]}。两包的相机时间相隔 **{number(s["source_dt_s"])}**，'
                  f'主机接收相隔 {number(s["receive_dt_s"])}。出口可能跳过中间帧，不能一概解释为“0.1 秒跳这么远”。',
                  f'  对应原始 VIO 三维变化 {number(s["raw_vio_step_3d_m"], "m")}，全局三维变化 '
                  f'{number(s["global_step_3d_m"], "m")}；三维量应互相比，不能直接与水平量相减。']
    if findings['largest_imu_lags']:
        item = findings['largest_imu_lags'][0]
        lines += [f'- **最大图像与最后已注册 IMU 的时间差：{number(item["lag_s"])}**，'
                  f'{item["time"]}，阶段为“{item["phase"]}”。这是传感器时间差，不是网络延迟。'
                  '即使发生于暂停/重定位期间，也应检查恢复后是否仍向 tracker 注册过旧 IMU，不能直接当作正常忽略。']
    checks = findings['tracker_checks']
    different = [x for x in checks if x['main'] != x['worker']]
    if different:
        lines += [f'- **tracker 对象线索：{len(checks)} 次可核对的主线程新建记录中，{len(different)} 次后续相机帧仍使用不同的 tracker ID。** '
                  '说明新建对象没有成为紧随其后的相机帧所用对象；需核对线程持有对象和重定位后的重置流程。它还不能单独证明漂移根因。']
    if findings['tracker_generations']:
        lines += ['','新版本由相机线程直接创建/切换 tracker，按代次核对：','',
                  '| 代次 | 切换时刻（UTC） | 已核对 VIO 帧数 | 对象不一致帧数 |','|---:|---|---:|---:|']
        for g in findings['tracker_generations']:
            lines.append(f'| {g["generation"]} | {g["time"]} | {g["frame_count"]} | {g["mismatch_count"]} |')
        lines += ['', '0 帧表示尚无恢复跟踪的证据，不能判定该次切换已验证。初始代次为 0，每次定位交接增加 1，失败/取消也会重建。',
                  '跨 tracker 代次的局部坐标位移留空，不计入漂移统计；全局出口是否突变仍单独检查。',
                  f'因 IMU 不新鲜、图像落后或请求取消而跳过的图像记录：{findings["imu_rejected_frames"]} 条；'
                  '原因和 tracker_role 见原始日志的 imu_frame_rejected。',
                  'vio_frames.csv 新增 track_cpu_s（当前调用线程 CPU 时间）、imu_wait_and_register_s、左右图像时间差；'
                  'CPU 时间不含其他线程，不能仅凭墙钟减 CPU 时间就认定是 GPU 等待。']
    lines += ['', '## 4. 关键数值怎么读', '', '| 指标 | 最大值 | 95% 的记录不超过 | 含义 |', '|---|---:|---:|---|']
    for metric, label, unit, explanation in [
        (summary['metrics']['send_to_receive_s'], '发送到接收', 'ms', '包含内核排队、接收线程调度；不等同纯网络延迟；仅同系统启动时钟可比较'),
        (summary['metrics']['receive_to_cache_s'], '接收到缓存更新', 'ms', '大时检查接收线程/缓存处理'),
        (summary['metrics']['frame_to_send_s'], 'SDK 取帧完成到发包', 'ms', '只统计实际发送的结果；过期丢弃的慢帧不在此项里'),
        (summary['vio_metrics']['track_s'], '上游 VIO 跟踪调用', 'ms', '包括未发送帧及暂停前后样本；须结合第 2 节时刻判断是否造成断流'),
        (summary['inference_duration_s'], 'SRU 推理耗时', 'ms', '单次尖峰可能是首次预热；不能仅凭它断定停车'),
        (summary['inference_end_pose_age_s'], '推理完成时，所用位姿的包龄', 's', '从该包缓存接收计时；本轮门槛 '+number(max_age))]:
        lines.append(f'| {label} | {number(metric["max"], unit)} | {number(metric["p95"], unit)} | {explanation} |')
    lines += ['', '数值统计覆盖本目录全部记录，不只行进阶段。`P95` 表示 95% 的记录不超过该值；最大值用于寻找偶发卡顿。',
              f'实际加载配置：位姿包龄门槛 {number(max_age)}；相邻位置变化基础门槛 '
              f'{number(recovery.get("max_position_jump_m"), "m")}；等待超时 {number(recovery.get("timeout_s"))}；'
              f'稳定窗口 {number(recovery.get("stable_s"))}。跨断流累计位移门槛还包含停车前运动余量，以具体事件为准。',
              '若目录中有多份 config，本节显示最后一份；各次配置原文保留在 summary.json，需分轮核对。', '',
              '## 5. 下一步先做什么', '']
    if not full:
        lines += ['1. 确认测试模式。要查下游时使用 `--sru`，并把机器人放在可重定位位置，等定位成功和 `running`。',
                  '2. 每轮重新 `prepare`、`source` 再启动；退出调度器后生成报告，确认采集时间更新。',
                  '3. 只有原始 VIO 时，只能先判断其坐标/时序；不要用“零条不一致”证明收发正常。']
    else:
        if summary['mismatches']:
            first_action = '**先核对坐标不一致。** 按 `summary.json → mismatches` 的源包序号，对照发送、接收和 SRU 输入，检查转换与状态组装。'
        elif any(w['upstream_gap'] for w in waits):
            first_action = '**先查自动等待对应时刻的上游卡顿。** 对照 `vio_frames.csv` 的 UTC 时刻、跟踪耗时、取帧间隔，以及 `imu_registration.csv` 的 IMU 时间差。'
        elif waits:
            first_action = '**先沿等待事件核对时序。** 相邻发送也变慢时查上游；发送连续但接收变慢时查接收调度/队列；接收连续而推理用旧包时查推理主循环。'
        elif errors:
            first_action = '**先核对第 2 节的 error 原因。** 位移超限需对照同两帧原始 VIO 与全局位姿，长断流需核对发包/收包时间；单看错误名称不足以判断根因。'
        else:
            first_action = '**本轮没有记录到自动位姿停车。** 先处理第 2 节列出的其他流程问题；若现场仍有异常，记录准确时刻后做下面的对照。'
        lines += ['1. '+first_action,
                  '2. **核对重定位后的 tracker 和 IMU 连续性。** 第 3 节若出现对象不一致或大 IMU 时间差，优先查此处；本报告没有替你修改运行逻辑。',
                  '3. **再验证是否漂移。** 单独做 `--no-sru` 静止对照约 30 秒，记录真实静止起止时间；再做已知距离/朝向的短段对照。'
                  '零速指令不等于机器人实际没动，不能据此直接判定 VIO 漂移。',
                  '4. **分别处理规划/定位失败。** A* 起点不可通行时核对重定位坐标在规划图的位置；`localize FAILED` 时核对视觉匹配与初始猜测。'
                  '不要把它们当成同一个“SRU 停机问题”，也不要仅靠放宽停车阈值掩盖原因。']
    lines += ['', '## 6. 数据完整性与证据文件', '',
              f'- 原始 VIO {findings["raw_vio_count"]} 帧；发送 {summary["tx_count"]} 包；接收 {summary["rx_count"]} 包；'
              f'精确匹配 {summary["matched_tx_rx"]} 包；推理开始 {findings["inference_count"]} 次。',
              f'- 有发送无接收 {summary["tx_without_rx"]}，有接收无发送 {summary["rx_without_tx"]}；'
              '可能受启动先后或日志缺失影响，不能直接认定网络丢包。',
              f'- 诊断队列丢行 {sum(summary["trace_dropped_by_logger"].values())}；损坏/半行 '
              f'{summary["malformed_or_partial_lines"]}；缺少正常收尾标记的进程 {len(summary["unclosed_loggers"])} 个。'
              '未收尾可能是仍在运行或强制退出，末尾证据可能缺失。',
              f'- 本目录记录到 {len({r.get("session_id") for r in events if r.get("session_id")})} 个调度会话。'
              '多次启动共用目录会混在一起；下次每轮重新 prepare。',
              '- [逐包坐标与时序](poses.csv)、[原始 VIO](vio_frames.csv)、[SRU 推理](inference.csv)、'
              '[IMU 注册](imu_registration.csv)、[机器可读完整统计](summary.json)。',
              '- 原始 `slam_*.jsonl` / `nav_*.jsonl` 保留逐事件证据。源相机时间只能在同设备时钟内求差，不能直接减 UTC。']
    for item in related:
        for field, label in [('session_file', '同会话任务信息'), ('failures_file', 'A* 失败记录'),
                             ('nav_file', '下游事件日志'), ('slam_file', 'SLAM 事件日志')]:
            if item.get(field):
                lines.append(f'- [{label}]({item[field]})')
    if not related:
        lines += ['- 未找到可通过 session_id 核实的同会话任务日志；A*、重定位失败、到达原因须另查调度器日志。']
    return '\n'.join(lines)+'\n'


def report(directory):
    rows, malformed = read_rows(directory)
    summary, samples, raw = analyze(rows)
    summary['malformed_or_partial_lines'] = malformed
    findings = interpret(rows, summary, samples)
    related = related_task_evidence(rows)
    summary['interpretation'] = findings
    summary['related_task_evidence'] = related
    write_json(directory/'summary.json', summary)
    write_csv(directory/'poses.csv', samples)
    write_csv(directory/'vio_frames.csv', raw)
    write_csv(directory/'imu_registration.csv', [{k:r.get(k) for k in
        ('source_timestamp_ns','mono_ns','registered','skipped_old','first_imu_ns','last_imu_ns','pending','image_to_last_imu_s')}
        for r in rows if r['stage']=='imu_register'])
    write_csv(directory/'inference.csv', [dict(stage=r['stage'], inference_id=r.get('inference_id'),
        source_sequence=r.get('source_sequence'), pose_sequence=r.get('pose_sequence'),
        source_timestamp_ns=r.get('source_timestamp_ns'), mono_ns=r['mono_ns'],
        receipt_age_s=delta(r['mono_ns'],r.get('received_ns')), duration_s=r.get('duration_s'),
        x=r.get('state_position',[None]*3)[0], y=r.get('state_position',[None]*3)[1],
        vx=r.get('linear_velocity',[None]*3)[0], vy=r.get('linear_velocity',[None]*3)[1],
        wz=r.get('angular_velocity',[None]*3)[2])
        for r in rows if r['stage'] in ('inference_start','inference_end','inference_error')])
    (directory/'report.md').write_text(render_report(directory, rows, summary, samples, findings, related), encoding='utf-8')
    print(f'报告：{directory/"report.md"}\n逐包：poses.csv；原始 VIO：vio_frames.csv；SRU 取样：inference.csv')
    return summary


def watch(directory):
    # Only tail trace files; never bind the SRU UDP port or connect to the robot.
    handles, pending, latest, prev = {}, {}, {}, {}
    try:
        while True:
            for path in directory.glob('*.jsonl'):
                if path not in handles:
                    handles[path] = path.open(encoding='utf-8')
                    pending[path] = ''
                lines = (pending[path]+handles[path].read()).split('\n')
                pending[path] = lines.pop()
                for line in lines:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row['stage'] in ('rx','inference_start','path_event','vio'):
                        stage = row['stage']
                        if stage == 'rx':
                            old = prev.get(row.get('stream_id',row['logger_id']))
                            row['receive_dt_s'] = delta(row.get('socket_received_ns'),old.get('socket_received_ns')) if old else None
                            prev[row.get('stream_id',row['logger_id'])] = row
                        latest[stage] = row
            row = latest.get('rx')
            if row:
                age = delta(time.monotonic_ns(),row.get('received_ns'))
                stamp = datetime.fromtimestamp(row['wall_ns']/1e9, timezone.utc).isoformat(timespec='milliseconds')
                print(f'RX {stamp} 源序号={row.get("source_sequence", "旧协议")} 相机ns={row.get("source_timestamp_ns")} '
                      f'间隔={row.get("receive_dt_s")}s 包龄={age:.3f}s '
                      f'XYZ原始={tuple(round(v,4) for v in row["wire_pose"][:3])} '
                      f'SRU序号={latest.get("inference_start",{}).get("source_sequence")} '
                      f'状态={latest.get("path_event",{}).get("path", "未收到")}',flush=True)
            else:
                v = latest.get('vio')
                if v:
                    print(f'VIO 相机ns={v["source_timestamp_ns"]} 局部XYZ={tuple(round(x,4) for x in v["local_position"])} '
                          f'特征数={v.get("observations")}；尚无下游接收记录。',flush=True)
                else:
                    print('等待诊断记录；脚本不会启动导航或占用 UDP 端口。',flush=True)
            time.sleep(1.)
    except KeyboardInterrupt:
        pass
    finally:
        for stream in handles.values():
            stream.close()
    report(directory)


def existing(slam_log, nav_log, out):
    out.mkdir(parents=True, exist_ok=True)
    samples, previous = [], None
    for line in slam_log.read_text(errors='replace').splitlines():
        if any(token in line for token in ('anchor=busy','anchor=ok','paused=1')):
            previous = None
        match = re.search(r'\[SCHED\] pose=\(([^)]+)\) t=([\d.]+)(?: source_timestamp_ns=(\d+))?',line)
        if not match:
            continue
        p = [float(v) for v in match[1].split(',')]
        row = dict(wall_s=float(match[2]), source_ns=int(match[3]) if match[3] else None,
                   x=p[0], y=p[1], z=p[2])
        if previous:
            row.update(wall_dt_s=row['wall_s']-previous['wall_s'], source_dt_s=delta(row['source_ns'],previous['source_ns']),
                       horizontal_step_m=math.dist([row['x'],row['z']],[previous['x'],previous['z']]))
        samples.append(row)
        previous = row
    events = [line for line in nav_log.read_text(errors='replace').splitlines()
              if any(f'path={state}' in line for state in ('error','waiting_pose','pose_resumed'))]
    largest = sorted([s for s in samples if 'horizontal_step_m' in s],key=lambda s:s['horizontal_step_m'],reverse=True)[:5]
    write_csv(out/'upstream_poses.csv',samples)
    result = dict(slam_log=str(slam_log),nav_log=str(nav_log),largest_steps=largest,events=events,
                  limitation='旧日志没有逐包发送/接收/SRU 取样关联；不能精确分解收发延迟。相邻 SCHED 事件可能跳过源帧。')
    write_json(out/'existing_summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    sub.add_parser('prepare').add_argument('--out')
    for name in ('watch','report'):
        p = sub.add_parser(name)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument('--latest',action='store_true')
        g.add_argument('--dir',type=Path)
    p = sub.add_parser('existing')
    for arg in ('slam-log','nav-log','out'):
        p.add_argument('--'+arg,type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out)
    elif args.command == 'existing':
        existing(args.slam_log,args.nav_log,args.out)
    else:
        directory = args.dir or Path((BASE/'latest.txt').read_text().strip())
        if not directory.is_dir():
            parser.error('诊断目录不存在；先运行 prepare')
        (watch if args.command=='watch' else report)(directory)


if __name__ == '__main__':
    main()
