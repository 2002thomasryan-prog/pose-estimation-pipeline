"""Dash dashboard for RunningScorer results, grouped by gait event. Port 8766."""
from __future__ import annotations
import base64, threading, time, webbrowser
import cv2, numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ALL, callback_context, dcc, html, no_update

PORT = 8766

C = {
    'header_bg':  '#1a2e1e', 'header_txt': '#e2e8f0', 'header_mut': '#8ea89a',
    'accent':     '#2f855a', 'good': '#38a169', 'good_bg': '#f0fff4',
    'warn':       '#d69e2e', 'warn_bg': '#fffff0', 'bad': '#e53e3e', 'bad_bg': '#fff5f5',
    'border':     '#e2e8f0', 'txt': '#2d3748', 'txt_light': '#718096',
    'white':      '#ffffff', 'panel_bg': '#f7f9fc',
}

EVENTS = [
    {'key':'td',  'label':'Touchdown (TD)',    'metrics':['trailing_knee_td','landing_foot_td','shin_ext_td','torso_tilt_td'], 'frame_src':'td_frames'},
    {'key':'ms',  'label':'Midstance (MS)',     'metrics':['thigh_gap_ms','sinking_ms'],                                       'frame_src':'ms_frames'},
    {'key':'to',  'label':'Toe-Off (TO)',       'metrics':['trailing_foot_to','thigh_sep_to','leading_ankle_to'],              'frame_src':'to_frames'},
    {'key':'mvp', 'label':'Max Velocity (MVP)', 'metrics':['thigh_sep_mvp','trailing_shin_mvp','leading_knee_mvp'],            'frame_src':'mvp_frames'},
]

_app_data: dict = {}
_app = None
_srv_thread = None


def set_data(scorer_dict: dict) -> None:
    _app_data.clear()
    _app_data.update(_json_safe(scorer_dict))


def launch(open_browser: bool = True) -> None:
    global _app, _srv_thread
    if _srv_thread is None or not _srv_thread.is_alive():
        _app = _build_app()
        _srv_thread = threading.Thread(
            target=lambda: _app.run(host='127.0.0.1', debug=False, port=PORT, use_reloader=False),
            daemon=True)
        _srv_thread.start()
    if open_browser:
        time.sleep(1.5)
        webbrowser.open(f'http://localhost:{PORT}')


def _json_safe(obj):
    if isinstance(obj, dict):          return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [_json_safe(x) for x in obj]
    if isinstance(obj, np.ndarray):    return _json_safe(obj.tolist())
    if isinstance(obj, np.integer):    return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj); return None if (v != v or v in (float('inf'), float('-inf'))) else v
    if isinstance(obj, np.bool_):      return bool(obj)
    if isinstance(obj, float):         return None if (obj != obj or obj in (float('inf'), float('-inf'))) else obj
    if isinstance(obj, (bool, int, str)) or obj is None: return obj
    return repr(obj)


def _read_frame_b64(video_path: str, frame_idx: int):
    if not video_path: return None
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read(); cap.release()
        if not ret: return None
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode('utf-8') if ok else None
    except Exception: return None


def _score_colour(score):
    if score is None: return C['txt_light']
    if score >= 75:   return C['good']
    if score >= 40:   return C['warn']
    return C['bad']


def _label_colour(label: str):
    return {'GREEN': C['good'], 'AMBER': C['warn'], 'RED': C['bad']}.get(label, C['txt_light'])


def _label_display(label: str):
    return {'GREEN': 'PASS', 'AMBER': 'WARN', 'RED': 'FAIL'}.get(label, label or '—')


def _event_score(ev_def, data):
    scores = data.get('param_scores', {})
    vals   = [scores[k] for k in ev_def['metrics'] if scores.get(k) is not None]
    return float(np.mean(vals)) if vals else None


def _middle_frame(ev_def, data):
    raw    = data.get('raw_traces', {})
    frames = raw.get(ev_def['frame_src'], [])
    if not frames: return 0
    f = frames[len(frames) // 2]
    return int(f) if f is not None else 0


def _fmt(val, unit):
    if val is None: return '—'
    try:
        f = float(val)
        if not np.isfinite(f): return '—'
        if unit == '%':                  return f'{f:.1f}%'
        if unit in ('score','ratio'):    return f'{f:.3f}'
        return f'{f:.1f}°'
    except: return str(val)


def _comic_panel(ev_def, data, selected):
    ekey   = ev_def['key']
    score  = _event_score(ev_def, data)
    colour = _score_colour(score)
    is_sel = ekey == selected

    fr  = _middle_frame(ev_def, data)
    b64 = _read_frame_b64(data.get('video_path') or '', fr)
    img = (html.Img(src=f'data:image/jpeg;base64,{b64}',
                    style={'width':'100%','height':'160px','objectFit':'cover',
                           'display':'block','borderRadius':'4px 4px 0 0'})
           if b64 else
           html.Div('No video', style={
               'height':'160px','display':'flex','alignItems':'center',
               'justifyContent':'center','color':C['txt_light'],'fontSize':'12px',
               'backgroundColor':'#edf2f7','borderRadius':'4px 4px 0 0'}))

    return html.Div(
        id={'type':'event-panel','index':ekey}, n_clicks=0,
        children=[
            html.Div(ev_def['label'], style={
                'fontSize':'12px','fontWeight':'700','color':C['white'],
                'textAlign':'center','padding':'6px 8px',
                'backgroundColor':colour,'borderRadius':'6px 6px 0 0'}),
            img,
            html.Div(f"Score: {score:.0f}" if score is not None else 'Score: —', style={
                'fontSize':'13px','fontWeight':'700','color':colour,
                'textAlign':'center','padding':'8px','backgroundColor':C['white'],
                'borderRadius':'0 0 6px 6px'}),
        ],
        style={
            'flex':'1','cursor':'pointer','border':f'3px solid {colour}','borderRadius':'8px',
            'boxShadow':('0 0 0 3px rgba(47,133,90,0.4)' if is_sel
                         else '0 2px 6px rgba(0,0,0,0.1)'),
            'overflow':'hidden','minWidth':'0'})


def _metric_row(param_key, data, selected_metric):
    meta   = next((p for p in data.get('params_meta',[]) if p['key']==param_key), {})
    report = data.get('emma_report',{}).get(param_key,{})
    label  = report.get('label','—')
    actual = report.get('actual')
    target = report.get('target','—')
    dev    = report.get('deviation')
    unit   = meta.get('unit','°')

    colour = _label_colour(label)
    bg     = (C['good_bg'] if label=='GREEN' else C['warn_bg'] if label=='AMBER' else C['bad_bg'])
    is_sel = param_key == selected_metric
    dev_txt = ''
    if dev is not None:
        try: dev_txt = f'{float(dev):+.1f}'
        except: pass

    return html.Div(
        id={'type':'metric-row','index':param_key}, n_clicks=0,
        children=[
            html.Div(meta.get('label',param_key),
                     style={'flex':'2','fontSize':'14px','fontWeight':'600','color':C['txt']}),
            html.Div(html.Span(_label_display(label), style={
                'display':'inline-block','padding':'2px 10px','borderRadius':'12px',
                'fontSize':'11px','fontWeight':'700','color':C['white'],
                'backgroundColor':colour,'minWidth':'48px','textAlign':'center'}),
                style={'flex':'1','display':'flex','alignItems':'center'}),
            html.Div(_fmt(actual,unit),
                     style={'flex':'1','fontSize':'13px','color':C['txt'],'fontFamily':'monospace'}),
            html.Div(str(target), style={'flex':'1','fontSize':'12px','color':C['txt_light']}),
            html.Div(dev_txt or '—',
                     style={'flex':'1','fontSize':'12px','color':colour,'fontFamily':'monospace'}),
        ],
        style={
            'display':'flex','alignItems':'center','gap':'12px','padding':'14px 20px',
            'backgroundColor': bg if is_sel else C['white'],
            'borderLeft':f'4px solid {colour}','borderBottom':f'1px solid {C["border"]}',
            'cursor':'pointer',
            'boxShadow':'inset 0 0 0 2px rgba(47,133,90,0.3)' if is_sel else 'none'})


def _cycle_bar_fig(param_key, data):
    records = data.get('cycle_records',{}).get(param_key,[])
    meta    = next((p for p in data.get('params_meta',[]) if p['key']==param_key), {})
    unit    = meta.get('unit','°')
    label   = meta.get('label',param_key)
    opt     = meta.get('optimal',[None,None])

    if not records:
        return go.Figure(layout=go.Layout(title=f'{label} — no cycle data',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'))

    actuals = [r.get('actual') for r in records]
    colours = [_label_colour(r.get('label','')) for r in records]
    cycles  = list(range(1, len(records)+1))
    fig     = go.Figure()

    if opt[0] is not None:
        fig.add_hrect(y0=opt[0], y1=opt[1], fillcolor='rgba(56,161,105,0.12)',
                      line_width=0, annotation_text=f'Target {opt[0]}–{opt[1]} {unit}',
                      annotation_position='top right',
                      annotation_font=dict(size=9, color=C['good']))

    fig.add_trace(go.Bar(
        x=cycles, y=[a if a is not None else 0 for a in actuals],
        marker_color=colours, marker_line_width=1.5,
        customdata=[r.get('label','') for r in records],
        hovertemplate=f'Cycle %{{x}}<br>{label}: %{{y:.2f}} {unit}<br>%{{customdata}}<extra></extra>'))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color=C['txt'], font_family='system-ui, sans-serif',
        margin=dict(l=50,r=20,t=40,b=40),
        title=dict(text=f'{label} — Per Cycle', font=dict(size=14)),
        showlegend=False, height=300)
    fig.update_xaxes(title_text='Cycle', tickvals=cycles, showgrid=False)
    fig.update_yaxes(title_text=unit, showgrid=True, gridcolor='rgba(0,0,0,0.07)')
    return fig


def _col_headers():
    return html.Div([
        html.Div(lbl, style={'flex':flex,'fontSize':'11px','fontWeight':'700',
                             'color':C['txt_light'],'textTransform':'uppercase','letterSpacing':'0.5px'})
        for lbl, flex in [('Metric','2'),('Result','1'),('Value','1'),('Target','1'),('Deviation','1')]
    ], style={'display':'flex','gap':'12px','padding':'10px 20px',
              'backgroundColor':'#edf2f7','borderBottom':f'2px solid {C["border"]}','flexShrink':'0'})


def _build_layout():
    data   = _app_data
    total  = data.get('total_score', 0) or 0
    colour = _score_colour(total)
    n_fr   = data.get('n_frames', 1) or 1
    comic  = [_comic_panel(ev, data, None) for ev in EVENTS]
    meta_txt = (f"{data.get('n_cycles',0)} cycles  ·  "
                f"{(data.get('backend') or '').upper()}  ·  "
                f"Side: {(data.get('dominant_side') or '?').title()}")

    return html.Div([
        dcc.Store(id='store-data',   data=data),
        dcc.Store(id='store-event',  data=None),
        dcc.Store(id='store-metric', data=None),
        dcc.Store(id='store-frame',  data=0),

        html.Div([
            html.Div('Running Biomechanical Assessment',
                     style={'fontWeight':'700','fontSize':'18px','color':C['header_txt']}),
            html.Div([html.Span(f'{total:.0f}/100',
                                style={'fontWeight':'800','fontSize':'20px','color':colour})],
                     style={'marginLeft':'auto'}),
        ], style={'backgroundColor':C['header_bg'],'padding':'12px 24px',
                  'display':'flex','alignItems':'center','flexShrink':'0'}),

        html.Div(id='breadcrumb', children='Running  ›  Session Overview',
                 style={'padding':'8px 24px','fontSize':'12px','color':C['txt_light'],
                        'borderBottom':f'1px solid {C["border"]}',
                        'backgroundColor':C['white'],'flexShrink':'0'}),

        html.Div(id='level1', children=[
            html.Div([
                html.Div(f'{total:.0f}',
                         style={'fontSize':'52px','fontWeight':'900','color':colour,'lineHeight':'1'}),
                html.Div('/100', style={'fontSize':'16px','color':C['txt_light'],'marginTop':'4px'}),
                html.Div(meta_txt, style={'marginTop':'8px','fontSize':'12px','color':C['txt_light']}),
            ], style={'display':'flex','alignItems':'baseline','gap':'8px',
                      'padding':'20px 28px 16px','flexShrink':'0'}),
        ], style={'overflowY':'auto','flex':'0 0 auto'}),

        html.Div(id='comic-strip', children=comic,
                 style={'display':'flex','gap':'16px','padding':'0 24px 24px','flexShrink':'0'}),

        html.Div(id='level2', style={'display':'none','flexDirection':'column','flex':'1','overflowY':'auto'},
                 children=[_col_headers(), html.Div(id='metric-table')]),

        html.Div(id='level3', style={'display':'none','flexDirection':'column','flex':'1','overflowY':'auto'}, children=[
            html.Div(id='metric-description',
                     style={'padding':'16px 24px','backgroundColor':C['white'],
                            'borderBottom':f'1px solid {C["border"]}','flexShrink':'0'}),
            html.Div([
                html.Div(dcc.Graph(id='cycle-bar-chart', config={'displayModeBar':False}),
                         style={'flex':'1','minWidth':'0'}),
                html.Div([
                    html.Div(id='jump-buttons',
                             style={'display':'flex','flexWrap':'wrap','gap':'6px',
                                    'padding':'12px','borderBottom':f'1px solid {C["border"]}'}),
                    html.Img(id='video-frame',
                             style={'width':'100%','maxHeight':'260px','objectFit':'contain','display':'block'}),
                    html.P(id='frame-label',
                           style={'textAlign':'center','fontSize':'11px','color':C['txt_light'],
                                  'margin':'4px 0 0 0','padding':'0 8px 8px'}),
                    html.Div([
                        html.P('Scrubber', style={'fontSize':'11px','color':C['txt_light'],'margin':'0 0 4px 0'}),
                        dcc.Slider(id='frame-scrubber', min=0, max=n_fr-1, step=1,
                                   value=0, marks={}, updatemode='drag',
                                   tooltip={'placement':'bottom'}),
                    ], style={'padding':'8px 12px 12px'}),
                ], style={'width':'340px','flexShrink':'0',
                          'borderLeft':f'1px solid {C["border"]}','backgroundColor':C['white']}),
            ], style={'display':'flex','flex':'1','minHeight':'0'}),
        ]),

    ], style={'display':'flex','flexDirection':'column','height':'100vh',
              'fontFamily':'system-ui, sans-serif','backgroundColor':C['panel_bg'],'overflow':'hidden'})


def _build_app():
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = _build_layout

    @app.callback(
        Output('store-event','data'),
        Output('store-metric','data'),
        Input({'type':'event-panel','index':ALL},'n_clicks'),
        State('store-event','data'),
        prevent_initial_call=True,
    )
    def select_event(_, current):
        t = callback_context.triggered_id
        if isinstance(t, dict) and t.get('type') == 'event-panel':
            return t['index'], None
        return current, no_update

    @app.callback(
        Output('store-metric','data', allow_duplicate=True),
        Input({'type':'metric-row','index':ALL},'n_clicks'),
        prevent_initial_call=True,
    )
    def select_metric(_):
        t = callback_context.triggered_id
        if isinstance(t, dict) and t.get('type') == 'metric-row':
            return t['index']
        return no_update

    @app.callback(
        Output('level1','style'),
        Output('level2','style'),
        Output('level3','style'),
        Output('breadcrumb','children'),
        Output('metric-table','children'),
        Output('comic-strip','children'),
        Output('comic-strip','style'),
        Input('store-event','data'),
        Input('store-metric','data'),
        State('store-data','data'),
    )
    def update_panels(event_key, metric_key, data):
        _HIDE      = {'display':'none'}
        _L1        = {'overflowY':'auto','flex':'0 0 auto'}
        _L2        = {'display':'flex','flexDirection':'column','flex':'1','overflowY':'auto'}
        _L3        = {'display':'flex','flexDirection':'column','flex':'1','overflowY':'auto'}
        _COMIC_VIS = {'display':'flex','gap':'16px','padding':'0 24px 24px','flexShrink':'0'}
        _COMIC_HID = {'display':'none'}
        data  = data or {}
        comic = [_comic_panel(ev, data, event_key) for ev in EVENTS]

        if not event_key:
            return _L1, _HIDE, _HIDE, 'Running  ›  Session Overview', [], comic, _COMIC_VIS

        ev_def   = next((e for e in EVENTS if e['key']==event_key), None)
        ev_label = ev_def['label'] if ev_def else event_key

        if not metric_key:
            crumb = [html.Span('Running',style={'cursor':'pointer','color':C['accent']}),
                     html.Span(f'  ›  {ev_label}',style={'color':C['txt']})]
            rows  = [_metric_row(mk, data, None) for mk in (ev_def['metrics'] if ev_def else [])]
            return _HIDE, _L2, _HIDE, crumb, rows, comic, _COMIC_HID

        meta    = next((p for p in data.get('params_meta',[]) if p['key']==metric_key), {})
        m_label = meta.get('label', metric_key)
        crumb   = [html.Span('Running',style={'cursor':'pointer','color':C['accent']}),
                   html.Span(f'  ›  {ev_label}',style={'cursor':'pointer','color':C['accent']}),
                   html.Span(f'  ›  {m_label}',style={'color':C['txt']})]
        rows    = [_metric_row(mk, data, metric_key) for mk in (ev_def['metrics'] if ev_def else [])]
        return _HIDE, _L2, _L3, crumb, rows, comic, _COMIC_HID

    @app.callback(
        Output('store-event','data', allow_duplicate=True),
        Output('store-metric','data', allow_duplicate=True),
        Input('breadcrumb','n_clicks'),
        State('store-event','data'),
        State('store-metric','data'),
        prevent_initial_call=True,
    )
    def breadcrumb_nav(_, event_key, metric_key):
        if metric_key: return event_key, None
        return None, None

    @app.callback(
        Output('cycle-bar-chart','figure'),
        Output('metric-description','children'),
        Output('jump-buttons','children'),
        Output('store-frame','data'),
        Input('store-metric','data'),
        State('store-data','data'),
        State('store-event','data'),
    )
    def update_level3(metric_key, data, event_key):
        if not metric_key or not data:
            return go.Figure(), [], [], 0
        meta    = next((p for p in data.get('params_meta',[]) if p['key']==metric_key), {})
        records = data.get('cycle_records',{}).get(metric_key,[])
        raw     = data.get('raw_traces',{})

        desc = [
            html.H4(meta.get('label',metric_key),
                    style={'margin':'0 0 4px 0','fontSize':'16px','color':C['txt']}),
            html.P(meta.get('description',''),
                   style={'margin':'0 0 4px 0','fontSize':'13px','color':C['txt_light'],'lineHeight':'1.5'}),
            html.P(f"Source: {meta.get('source','—')}",
                   style={'margin':'0','fontSize':'11px','color':C['txt_light'],'fontStyle':'italic'}),
        ]

        ef_key   = meta.get('event_frames_key') or 'td_frames'
        ef_list  = raw.get(ef_key, [])
        ev_def   = next((e for e in EVENTS if e['key']==(event_key or '')), None)
        fallback = raw.get(ev_def['frame_src'] if ev_def else 'td_frames', [])

        buttons = []
        for ci, rec in enumerate(records):
            if ci < len(ef_list) and ef_list[ci] is not None: fr = int(ef_list[ci])
            elif ci < len(fallback) and fallback[ci] is not None: fr = int(fallback[ci])
            else: fr = 0
            colour = _label_colour(rec.get('label',''))
            buttons.append(html.Button(f'C{ci+1}',
                id={'type':'jump-btn','index':fr}, n_clicks=0,
                style={'padding':'4px 10px','fontSize':'11px','border':f'2px solid {colour}',
                       'borderRadius':'4px','cursor':'pointer','backgroundColor':colour,
                       'color':C['white'],'fontWeight':'700'}))

        mid_fr = 0
        src = ef_list if ef_list else fallback
        if src:
            f = src[len(src)//2]
            mid_fr = int(f) if f is not None else 0
        return _cycle_bar_fig(metric_key, data), desc, buttons, mid_fr

    @app.callback(
        Output('store-frame','data', allow_duplicate=True),
        Input({'type':'jump-btn','index':ALL},'n_clicks'),
        prevent_initial_call=True,
    )
    def jump(_):
        t = callback_context.triggered_id
        if isinstance(t, dict) and t.get('type') == 'jump-btn': return int(t['index'])
        return no_update

    @app.callback(
        Output('store-frame','data', allow_duplicate=True),
        Input('frame-scrubber','value'),
        prevent_initial_call=True,
    )
    def scrub(val):
        return int(val) if val is not None else no_update

    @app.callback(
        Output('frame-scrubber','value', allow_duplicate=True),
        Input('store-frame','data'),
        prevent_initial_call=True,
    )
    def sync_scrubber(fi):
        return fi if fi is not None else no_update

    @app.callback(
        Output('video-frame','src'),
        Output('frame-label','children'),
        Input('store-frame','data'),
        State('store-data','data'),
    )
    def render_frame(fi, data):
        if not data or fi is None: return '', ''
        path = data.get('video_path') or ''
        n_fr = data.get('n_frames',1) or 1
        fps  = data.get('fps',30) or 30
        b64  = _read_frame_b64(path, fi) if path else None
        return (f'data:image/jpeg;base64,{b64}' if b64 else ''), \
               f'Frame {fi} / {n_fr-1}  ·  t = {fi/fps:.2f} s'

    return app
