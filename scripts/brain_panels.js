/* Shared canvas drawing primitives for Brain Explorer and future frame exports. */
(function () {
  const COLORS = {
    bg: '#0a111e', grid: '#29384e', ink: '#e8f1fb', muted: '#9aacbf',
    cyan: '#42d9e8', violet: '#a78bfa', green: '#5ee0a0', pink: '#fb7185', amber: '#fbbf24'
  };
  const dpr = () => window.devicePixelRatio || 1;
  function prepare(canvas, minHeight = 240) {
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(240, rect.width || canvas.clientWidth || 500);
    const height = Math.max(minHeight, rect.height || canvas.clientHeight || minHeight);
    canvas.width = Math.round(width * dpr());
    canvas.height = Math.round(height * dpr());
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr(), 0, 0, dpr(), 0, 0);
    ctx.clearRect(0, 0, width, height);
    return { ctx, width, height };
  }
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
  function diverging(value, maximum) {
    const t = clamp(value / Math.max(maximum, 1e-9), -1, 1);
    const neutral = [16, 24, 38], positive = [74, 222, 190], negative = [251, 113, 133];
    const target = t < 0 ? negative : positive, q = Math.abs(t) ** 0.72;
    return `rgb(${neutral.map((v, i) => Math.round(v + (target[i] - v) * q)).join(',')})`;
  }
  function scalarColor(t) {
    const stops = [[36, 72, 142], [44, 160, 170], [154, 213, 105], [249, 189, 74], [225, 89, 57]];
    const v = clamp(t, 0, 1) * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(v)), f = v - i;
    return `rgb(${stops[i].map((c, k) => Math.round(c + (stops[i + 1][k] - c) * f)).join(',')})`;
  }
  function drawNetwork(canvas, model, topK = 300, activations = null) {
    const { ctx, width: w, height: h } = prepare(canvas, 520);
    ctx.fillStyle = COLORS.bg; ctx.fillRect(0, 0, w, h);
    const layers = [model.inputs.length, model.hidden_sizes[0], model.hidden_sizes[1], model.outputs.length];
    const left = 142, right = 72, xs = [left, left + (w-left-right)/3, left + 2*(w-left-right)/3, w-right];
    const yAt = (layer, index) => 34 + index * (h - 68) / Math.max(1, layers[layer] - 1);
    const nodes = [];
    for (let layer = 0; layer < 4; layer++) {
      nodes[layer] = Array.from({ length: layers[layer] }, (_, index) => ({ x: xs[layer], y: yAt(layer, index) }));
    }
    const edges = [];
    for (let layer = 0; layer < 3; layer++) {
      const matrix = model.weights[layer];
      matrix.forEach((row, i) => row.forEach((weight, j) => edges.push({ layer, i, j, weight })));
    }
    edges.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
    const selected = edges.slice(0, Math.max(0, Math.min(edges.length, topK)));
    const activationFor = layer => {
      if (!activations) return null;
      if (activations.length >= 4) return activations[layer];
      return layer === 0 ? null : activations[layer - 1];
    };
    const magnitudeFor = edge => {
      const source = activationFor(edge.layer);
      return Math.abs(edge.weight) * (source && edge.i < source.length ? Math.abs(source[edge.i]) : 1);
    };
    const maximum = Math.max(1e-8, ...selected.map(magnitudeFor));
    ctx.lineCap = 'round';
    for (const edge of selected) {
      const a = nodes[edge.layer][edge.i], b = nodes[edge.layer + 1][edge.j];
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
      ctx.strokeStyle = edge.weight >= 0 ? '#57d9c2' : '#fb7185';
      const signal = magnitudeFor(edge) / maximum;
      ctx.globalAlpha = 0.10 + 0.78 * signal;
      ctx.lineWidth = 0.35 + 2.5 * signal;
      ctx.stroke();
      edge._a = a; edge._b = b;
    }
    ctx.globalAlpha = 1;
    const activ = activations || [];
    for (let layer = 0; layer < 4; layer++) {
      nodes[layer].forEach((point, index) => {
        const value = activations?.length >= 4 ? activ[layer]?.[index]
          : layer === 1 ? activ[0]?.[index] : layer === 2 ? activ[1]?.[index] : layer === 3 ? activ[2]?.[index] : null;
        const intensity = value == null ? 0 : Math.abs(value);
        ctx.beginPath(); ctx.arc(point.x, point.y, layer === 0 || layer === 3 ? 4.2 : 4.8, 0, Math.PI * 2);
        ctx.fillStyle = value == null ? '#d3e1ef' : (value >= 0 ? COLORS.cyan : COLORS.pink);
        ctx.globalAlpha = 0.34 + 0.66 * intensity; ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 11 * intensity;
        ctx.fill(); ctx.shadowBlur = 0; ctx.globalAlpha = 1;
        if (layer === 0) {
          ctx.fillStyle = COLORS.muted; ctx.font = '10px Segoe UI'; ctx.textAlign = 'right';
          ctx.fillText(model.inputs[index], point.x - 9, point.y + 3);
        } else if (layer === 3) {
          ctx.fillStyle = COLORS.ink; ctx.font = '11px Segoe UI'; ctx.textAlign = 'left';
          ctx.fillText(model.outputs[index], point.x + 10, point.y + 4);
        }
      });
    }
    ctx.textAlign = 'center'; ctx.fillStyle = COLORS.ink; ctx.font = '600 12px Segoe UI';
    ['Inputs', 'Hidden 1 · 24', 'Hidden 2 · 24', 'Actions'].forEach((name, i) => ctx.fillText(name, xs[i], 17));
    canvas._networkEdges = selected;
    canvas.onmousemove = event => {
      const rect = canvas.getBoundingClientRect(), mx = event.clientX - rect.left, my = event.clientY - rect.top;
      let best = null, distance = 7;
      for (const edge of selected) {
        const { _a: a, _b: b } = edge, dx = b.x-a.x, dy = b.y-a.y;
        const t = clamp(((mx-a.x)*dx+(my-a.y)*dy)/(dx*dx+dy*dy+1e-8), 0, 1);
        const d = Math.hypot(mx-(a.x+t*dx), my-(a.y+t*dy));
        if (d < distance) { distance = d; best = edge; }
      }
      canvas.title = best ? `${['input → hidden 1','hidden 1 → hidden 2','hidden 2 → output'][best.layer]} · weight ${best.weight.toFixed(5)}` : 'Hover a visible connection to inspect its exact weight';
    };
  }
  function drawMatrix(canvas, matrix, bias, title, rowLabel, colLabel) {
    const rows = matrix.length, cols = matrix[0]?.length || 0;
    const { ctx, width: w, height: h } = prepare(canvas, 220);
    ctx.fillStyle = COLORS.bg; ctx.fillRect(0, 0, w, h);
    const margin = { l: 58, r: 12, t: 26, b: 36 }, cw = (w-margin.l-margin.r)/cols, ch = (h-margin.t-margin.b)/rows;
    const max = Math.max(1e-8, ...matrix.flat().map(v => Math.abs(v)));
    ctx.font = '10px Segoe UI'; ctx.fillStyle = COLORS.ink; ctx.textAlign = 'left'; ctx.fillText(title, 6, 15);
    matrix.forEach((row, i) => row.forEach((value, j) => {
      ctx.fillStyle = diverging(value, max); ctx.fillRect(margin.l+j*cw, margin.t+i*ch, Math.ceil(cw), Math.ceil(ch));
    }));
    ctx.fillStyle = COLORS.muted; ctx.textAlign = 'center'; ctx.fillText(colLabel, margin.l+(w-margin.l-margin.r)/2, h-5);
    ctx.save(); ctx.translate(12, margin.t+(h-margin.t-margin.b)/2); ctx.rotate(-Math.PI/2); ctx.fillText(rowLabel, 0, 0); ctx.restore();
    const biasCanvas = canvas.parentElement.querySelector('canvas.bias');
    if (biasCanvas) drawVector(biasCanvas, bias, 'Bias · symmetric around 0');
  }
  function drawVector(canvas, values, title) {
    const { ctx, width: w, height: h } = prepare(canvas, 90);
    ctx.fillStyle = COLORS.bg; ctx.fillRect(0, 0, w, h);
    const max = Math.max(1e-8, ...values.map(v => Math.abs(v))), gap = 1, cell = Math.max(1, (w-8)/values.length-gap);
    ctx.fillStyle = COLORS.muted; ctx.font = '10px Segoe UI'; ctx.fillText(title, 5, 13);
    values.forEach((v, i) => { ctx.fillStyle = diverging(v, max); ctx.fillRect(4+i*(cell+gap), 25, cell, h-30); });
  }
  function drawWeightMatrices(container, model) {
    container.innerHTML = '';
    const titles = ['Input → hidden 1', 'Hidden 1 → hidden 2', 'Hidden 2 → output'];
    model.weights.forEach((matrix, i) => {
      const section = document.createElement('div'); section.className = 'weightSlice';
      const title = document.createElement('h3'); title.textContent = titles[i];
      const heat = document.createElement('canvas'); heat.className = 'weightHeat';
      const bias = document.createElement('canvas'); bias.className = 'bias';
      section.append(title, heat, bias); container.appendChild(section);
      drawMatrix(heat, matrix, model.biases[i], titles[i], i === 0 ? 'inputs' : `layer ${i}`, i === 2 ? 'actions' : `layer ${i+1}`);
    });
  }
  function drawBars(canvas, values, labels, title, color = null, fixedMax = null) {
    const { ctx, width: w, height: h } = prepare(canvas, 190);
    ctx.fillStyle = COLORS.bg; ctx.fillRect(0, 0, w, h);
    const pad = { l: 42, r: 12, t: 20, b: 36 }, max = fixedMax || Math.max(1e-8, ...values.map(v => Math.abs(v)));
    const zero = pad.t + (h - pad.t - pad.b) / 2, amplitude = (h - pad.t - pad.b) / 2;
    ctx.strokeStyle = COLORS.grid; ctx.fillStyle = COLORS.muted; ctx.font = '10px Segoe UI';
    ctx.beginPath(); ctx.moveTo(pad.l, zero); ctx.lineTo(w-pad.r,zero); ctx.stroke();
    ctx.fillStyle = COLORS.ink; ctx.fillText(title, pad.l, 12);
    const step=(w-pad.l-pad.r)/values.length, bar=Math.max(2,step*.68);
    values.forEach((v,i)=>{const x=pad.l+i*step+(step-bar)/2, height=Math.abs(v)/max*amplitude;ctx.fillStyle=color?(Array.isArray(color)?color[i%color.length]:color):diverging(v,max);ctx.globalAlpha=.56+.44*Math.min(1,Math.abs(v)/max);ctx.fillRect(x,v<0?zero:zero-height,bar,height);ctx.globalAlpha=1;ctx.fillStyle=COLORS.muted;ctx.save();ctx.translate(x+bar/2,h-pad.b+7);ctx.rotate(-Math.PI/3);ctx.textAlign='right';ctx.fillText(labels[i],0,0);ctx.restore()});
    ctx.textAlign='left'; ctx.fillStyle=COLORS.muted; ctx.fillText(`+${max.toFixed(2)}`,2,pad.t+3); ctx.fillText('0',pad.l-12,zero+3); ctx.fillText(`−${max.toFixed(2)}`,2,h-pad.b+3);
  }
  function drawDecision(canvas, data) {
    const values=data.values, rows=values.length, cols=values[0].length, min=Math.min(...values.flat()), max=Math.max(...values.flat());
    const {ctx,width:w,height:h}=prepare(canvas,330);ctx.fillStyle=COLORS.bg;ctx.fillRect(0,0,w,h);
    const p={l:62,r:14,t:14,b:42},cw=(w-p.l-p.r)/cols,ch=(h-p.t-p.b)/rows;
    const center=data.output==='speed'?132.5:0,scale=Math.max(1e-8,...values.flat().map(v=>Math.abs(v-center)));
    values.forEach((row,y)=>row.forEach((v,x)=>{ctx.fillStyle=diverging(v-center,scale);ctx.fillRect(p.l+x*cw,p.t+(rows-1-y)*ch,Math.ceil(cw),Math.ceil(ch))}));
    ctx.strokeStyle=COLORS.grid;ctx.lineWidth=1;ctx.strokeRect(p.l,p.t,w-p.l-p.r,h-p.t-p.b);ctx.fillStyle=COLORS.muted;ctx.font='11px Segoe UI';ctx.textAlign='center';ctx.fillText(`${data.x_name} (input ${data.x_index})`,p.l+(w-p.l-p.r)/2,h-8);ctx.save();ctx.translate(15,p.t+(h-p.t-p.b)/2);ctx.rotate(-Math.PI/2);ctx.fillText(`${data.y_name} (input ${data.y_index})`,0,0);ctx.restore();ctx.textAlign='left';ctx.fillText(`${min.toFixed(2)}–${max.toFixed(2)} ${data.unit}`,p.l,p.t-3);
  }
  function drawPitch(canvas, points) {
    const {ctx,width:w,height:h}=prepare(canvas,260);ctx.fillStyle=COLORS.bg;ctx.fillRect(0,0,w,h);const p={l:48,r:16,t:18,b:42};
    const X=x=>p.l+clamp(x/14,0,1)*(w-p.l-p.r),Y=y=>h-p.b-(clamp((y+1)/2,0,1))*(h-p.t-p.b);
    ctx.fillStyle='#17364a';ctx.globalAlpha=.50;ctx.fillRect(X(5),p.t,X(8)-X(5),h-p.t-p.b);ctx.globalAlpha=1;
    ctx.strokeStyle=COLORS.grid;ctx.fillStyle=COLORS.muted;ctx.font='10px Segoe UI';ctx.textAlign='right';
    for(let i=0;i<=7;i++){const x=X(i*2);ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,h-p.b);ctx.stroke();ctx.fillText(String(i*2),x+3,h-p.b+13)}
    for(let i=0;i<=4;i++){const v=-1+i*.5,y=Y(v);ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke();ctx.fillText(v.toFixed(1),p.l-6,y+3)}
    // Batsman's stumps and popping crease are at zero on the bounce-distance axis.
    ctx.strokeStyle='#e2e8f0';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(X(0),Y(-.34));ctx.lineTo(X(0),Y(.34));ctx.stroke();
    ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(X(0),p.t);ctx.lineTo(X(0),h-p.b);ctx.stroke();
    ctx.setLineDash([4,3]);ctx.strokeStyle='#d5e3ef';ctx.beginPath();ctx.moveTo(X(.45),p.t);ctx.lineTo(X(.45),h-p.b);ctx.stroke();ctx.setLineDash([]);
    const outcomes={runs:COLORS.green,dot:'#8094aa',wide:COLORS.amber,no_ball:'#fb923c',bowled:COLORS.pink,lbw:'#f43f5e',caught_behind:COLORS.violet,caught_field:'#ef4444'};
    (points||[]).forEach((point,i)=>{ctx.beginPath();ctx.arc(X(point.length_m||0),Y(point.line_m||0),i===point.current?6:4,0,Math.PI*2);ctx.fillStyle=outcomes[point.result]||COLORS.cyan;ctx.globalAlpha=i===point.current?1:.72;ctx.fill();if(i===point.current){ctx.strokeStyle='#fff';ctx.lineWidth=1.5;ctx.stroke()}ctx.fillStyle=COLORS.ink;ctx.font='9px Segoe UI';ctx.textAlign='center';ctx.fillText(String(i+1),X(point.length_m||0),Y(point.line_m||0)-7)});ctx.globalAlpha=1;
    ctx.fillStyle=COLORS.muted;ctx.textAlign='center';ctx.font='10px Segoe UI';ctx.fillText('bounce distance from stumps (m) · shaded = good length · dashed = popping crease',p.l+(w-p.l-p.r)/2,h-5);ctx.save();ctx.translate(10,p.t+(h-p.t-p.b)/2);ctx.rotate(-Math.PI/2);ctx.fillText('line offset (m)',0,0);ctx.restore();
    ctx.font='9px Segoe UI';ctx.textAlign='left';let lx=p.l+8;for(const [name,color] of [['runs',COLORS.green],['dot','#8094aa'],['wicket',COLORS.pink],['extra',COLORS.amber]]){ctx.beginPath();ctx.arc(lx,p.t+9,3.5,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();ctx.fillStyle=COLORS.muted;ctx.fillText(name,lx+7,p.t+12);lx+=name.length*5.5+24}
  }
  function drawTrajectory(canvas, points, timeSeconds = 0) {
    const {ctx,width:w,height:h}=prepare(canvas,220);ctx.fillStyle=COLORS.bg;ctx.fillRect(0,0,w,h);
    const p={l:42,r:14,t:16,b:28},xMin=-2.4,xMax=2.4,zMax=3.2;
    const X=x=>p.l+clamp((x-xMin)/(xMax-xMin),0,1)*(w-p.l-p.r),Y=z=>h-p.b-clamp(z/zMax,0,1)*(h-p.t-p.b);
    ctx.strokeStyle=COLORS.grid;ctx.fillStyle=COLORS.muted;ctx.font='10px Segoe UI';
    for(let i=0;i<=4;i++){const x=p.l+i*(w-p.l-p.r)/4;ctx.beginPath();ctx.moveTo(x,p.t);ctx.lineTo(x,h-p.b);ctx.stroke();ctx.textAlign='center';ctx.fillText((xMin+i*(xMax-xMin)/4).toFixed(1),x,h-7)}
    for(let i=0;i<=3;i++){const z=i;const y=Y(z);ctx.beginPath();ctx.moveTo(p.l,y);ctx.lineTo(w-p.r,y);ctx.stroke();ctx.textAlign='right';ctx.fillText(`${z} m`,p.l-5,y+3)}
    const flown=(points||[]).filter(point=>Number(point.t_s)<=timeSeconds);if(flown.length>1){ctx.beginPath();flown.forEach((point,index)=>index?ctx.lineTo(X(point.x_m),Y(point.z_m)):ctx.moveTo(X(point.x_m),Y(point.z_m)));ctx.strokeStyle=COLORS.cyan;ctx.lineWidth=2.2;ctx.stroke()}
    if(flown.length){const point=flown.at(-1),x=X(point.x_m),y=Y(point.z_m);ctx.beginPath();ctx.arc(x,y,5,0,Math.PI*2);ctx.fillStyle=COLORS.ink;ctx.fill();ctx.strokeStyle=COLORS.cyan;ctx.lineWidth=2;ctx.stroke()}
    ctx.textAlign='center';ctx.fillStyle=COLORS.muted;ctx.fillText(`flight path · ${timeSeconds.toFixed(3)} s`,p.l+(w-p.l-p.r)/2,12);
  }
  function drawSensitivity(canvas, data) {
    const {ctx,width:w,height:h}=prepare(canvas,360);ctx.fillStyle=COLORS.bg;ctx.fillRect(0,0,w,h);const p={l:142,r:18,t:22,b:32};
    const rows=data.inputs.length,rowH=(h-p.t-p.b)/rows,max=Math.max(1e-8,...data.scores.flat());ctx.font='10px Segoe UI';ctx.textAlign='right';
    data.inputs.forEach((name,i)=>{let y=p.t+i*rowH;ctx.fillStyle=COLORS.muted;ctx.fillText(name,p.l-8,y+rowH*.68);data.scores.forEach((series,o)=>{let barH=Math.min(5,rowH/5),offset=(o-1.5)*barH*1.15;ctx.fillStyle=[COLORS.cyan,COLORS.violet,COLORS.green,COLORS.amber][o];ctx.fillRect(p.l,y+rowH*.5+offset, (w-p.l-p.r)*series[i]/max,barH)})});
    ctx.textAlign='left';ctx.fillStyle=COLORS.muted;ctx.fillText('permutation importance · mean squared action change',p.l,h-8);ctx.fillText(max.toFixed(4),p.l,p.t-7);
  }
  function drawPopulation(canvas, data) {
    const {ctx,width:w,height:h}=prepare(canvas,330);ctx.fillStyle=COLORS.bg;ctx.fillRect(0,0,w,h);const p={l:48,r:18,t:18,b:38},pts=data.points||[],xs=pts.map(p=>p[0]),ys=pts.map(p=>p[1]);if(data.champion){xs.push(data.champion[0]);ys.push(data.champion[1])}
    if(!pts.length){ctx.fillStyle=COLORS.muted;ctx.fillText(data.error||'Population snapshot is not available',p.l,h/2);return}
    const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys),X=x=>p.l+(x-x0)/(x1-x0||1)*(w-p.l-p.r),Y=y=>h-p.b-(y-y0)/(y1-y0||1)*(h-p.t-p.b),fit=data.fitness||[],f0=Math.min(...fit),f1=Math.max(...fit);
    ctx.strokeStyle=COLORS.grid;ctx.beginPath();ctx.moveTo(p.l,h-p.b);ctx.lineTo(w-p.r,h-p.b);ctx.moveTo(p.l,p.t);ctx.lineTo(p.l,h-p.b);ctx.stroke();
    const ancestry=new Set(data.ancestry_ids||[]),ids=data.genome_ids||[];
    pts.forEach((point,i)=>{ctx.beginPath();ctx.arc(X(point[0]),Y(point[1]),3.2,0,Math.PI*2);ctx.fillStyle=scalarColor((fit[i]-f0)/(f1-f0||1));ctx.globalAlpha=.78;ctx.fill();if(ancestry.has(ids[i])){ctx.beginPath();ctx.arc(X(point[0]),Y(point[1]),7,0,Math.PI*2);ctx.strokeStyle=COLORS.amber;ctx.lineWidth=2;ctx.stroke()}});ctx.globalAlpha=1;
    if(data.champion){ctx.fillStyle=COLORS.amber;ctx.strokeStyle='#fff4c2';ctx.lineWidth=1.5;ctx.beginPath();const cx=X(data.champion[0]),cy=Y(data.champion[1]);for(let i=0;i<10;i++){const r=i%2?5:10,a=-Math.PI/2+i*Math.PI/5;const px=cx+Math.cos(a)*r,py=cy+Math.sin(a)*r;i?ctx.lineTo(px,py):ctx.moveTo(px,py)}ctx.closePath();ctx.fill();ctx.stroke()}
    ctx.fillStyle=COLORS.muted;ctx.textAlign='center';ctx.font='11px Segoe UI';ctx.fillText(`PC1 · ${(100*data.explained_variance[0]).toFixed(1)}% explained`,p.l+(w-p.l-p.r)/2,h-8);ctx.save();ctx.translate(12,p.t+(h-p.t-p.b)/2);ctx.rotate(-Math.PI/2);ctx.fillText(`PC2 · ${(100*data.explained_variance[1]).toFixed(1)}% explained`,0,0);ctx.restore();ctx.textAlign='left';ctx.fillText(`gen ${data.snapshot_generation} · each dot is one genome`,p.l,p.t-4);
  }
  function drawStats(canvas, data) {
    const {ctx,width:w,height:h}=prepare(canvas,420);ctx.fillStyle=COLORS.bg;ctx.fillRect(0,0,w,h);
    const points=data.points||[],colors=[COLORS.cyan,COLORS.violet,COLORS.green];
    if(!points.length){ctx.fillStyle=COLORS.muted;ctx.font='12px Segoe UI';ctx.fillText('Checkpoint statistics unavailable',24,h/2);return}
    const metrics=[
      {key:'mean_abs_weight',title:'Mean |weight|'},
      {key:'std_weight',title:'Weight standard deviation'},
      {key:'saturated_hidden_share',title:'Saturated hidden activations (%)',scale:100},
      {key:'l2_from_previous',title:'L2 distance from previous checkpoint'}
    ];
    const padX=38,gapX=36,gapY=28,top=38,bottom=24,cellW=(w-2*padX-gapX)/2,cellH=(h-top-bottom-gapY)/2;
    const xAt=i=>padX+i*(cellW-18)/Math.max(1,points.length-1);
    metrics.forEach((metric,m)=>{
      const col=m%2,row=Math.floor(m/2),x0=padX+col*(cellW+gapX),y0=top+row*(cellH+gapY);
      const plot={l:x0+34,r:x0+cellW-7,t:y0+22,b:y0+cellH-24};
      const series=points.flatMap(point=>point.layers.map(layer=>layer[metric.key]).filter(v=>v!=null).map(v=>v*(metric.scale||1)));
      ctx.fillStyle=COLORS.ink;ctx.font='600 11px Segoe UI';ctx.textAlign='left';ctx.fillText(metric.title,x0,y0+11);
      if(!series.length){ctx.fillStyle=COLORS.muted;ctx.font='10px Segoe UI';ctx.fillText('No values available',plot.l,plot.t+18);return}
      let lo=Math.min(...series),hi=Math.max(...series);if(metric.key==='saturated_hidden_share'){lo=0;hi=Math.max(5,hi)}else{const delta=(hi-lo)||Math.max(Math.abs(hi)*.05,.01);lo=Math.max(0,lo-delta*.08);hi+=delta*.08}
      const yAt=value=>plot.b-(value-lo)/(hi-lo||1)*(plot.b-plot.t);
      ctx.strokeStyle=COLORS.grid;ctx.lineWidth=1;ctx.font='9px Segoe UI';ctx.fillStyle=COLORS.muted;
      for(let tick=0;tick<=2;tick++){const y=plot.t+tick*(plot.b-plot.t)/2;ctx.beginPath();ctx.moveTo(plot.l,y);ctx.lineTo(plot.r,y);ctx.stroke();ctx.textAlign='right';const value=hi-(hi-lo)*tick/2;ctx.fillText(metric.key==='saturated_hidden_share'?`${value.toFixed(0)}%`:value.toFixed(3),plot.l-5,y+3)}
      ctx.beginPath();ctx.moveTo(plot.l,plot.t);ctx.lineTo(plot.l,plot.b);ctx.stroke();
      for(let layer=0;layer<3;layer++){
        let active=false;ctx.beginPath();
        points.forEach((point,i)=>{const raw=point.layers[layer][metric.key];if(raw==null){active=false;return}const value=raw*(metric.scale||1),x=plot.l+i*(plot.r-plot.l)/Math.max(1,points.length-1),y=yAt(value);if(active)ctx.lineTo(x,y);else ctx.moveTo(x,y);active=true});
        ctx.strokeStyle=colors[layer];ctx.lineWidth=1.8;ctx.stroke();
      }
      ctx.fillStyle=COLORS.muted;ctx.font='9px Segoe UI';ctx.textAlign='left';ctx.fillText(`gen ${points[0].generation}`,plot.l,plot.b+17);ctx.textAlign='right';ctx.fillText(`gen ${points.at(-1).generation}`,plot.r,plot.b+17);ctx.textAlign='left';
    });
    const legend=[['input → hidden 1',colors[0]],['hidden 1 → hidden 2',colors[1]],['hidden 2 → output',colors[2]]];let lx=padX;
    ctx.font='9px Segoe UI';legend.forEach(([label,color])=>{ctx.fillStyle=color;ctx.fillRect(lx,17,9,3);ctx.fillStyle=COLORS.muted;ctx.fillText(label,lx+13,21);lx+=label.length*5.4+34});
  }
  function drawFamily(canvas, data, windowSize=1500) {
    const {ctx,width:w,height:h}=prepare(canvas,340);ctx.fillStyle=COLORS.bg;ctx.fillRect(0,0,w,h);
    if(!data.available||!data.nodes?.length){ctx.fillStyle=COLORS.muted;ctx.fillText(data.message||'Lineage is not available',18,h/2);return}
    const nodes=data.nodes,edges=data.edges||[],maxGen=data.champion_generation||Math.max(...nodes.map(n=>n.generation)),minGen=Math.max(0,maxGen-windowSize),visible=nodes.filter(n=>n.generation>=minGen);const p={l:38,r:18,t:18,b:30},X=g=>p.l+(g-minGen)/Math.max(1,maxGen-minGen)*(w-p.l-p.r),primary=visible.filter(n=>!n.donor).sort((a,b)=>a.generation-b.generation),yMain=h*.48;
    const byId=new Map();primary.forEach(n=>byId.set(n.id,{x:X(n.generation),y:yMain}));visible.filter(n=>n.donor).forEach(n=>byId.set(n.id,{x:X(n.generation),y:yMain+54}));
    edges.forEach(edge=>{const a=byId.get(edge.from),b=byId.get(edge.to);if(!a||!b)return;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=edge.kind==='crossover'?COLORS.violet:'#4e6680';ctx.lineWidth=edge.kind==='crossover'?1.2:.8;if(edge.kind==='crossover')ctx.setLineDash([4,3]);ctx.stroke();ctx.setLineDash([])});
    visible.forEach(node=>{const q=byId.get(node.id);if(!q)return;ctx.beginPath();ctx.arc(q.x,q.y,node.donor?2.1:2.8,0,Math.PI*2);ctx.fillStyle=node.was_elite?COLORS.amber:scalarColor(clamp((node.fitness+1)/2,0,1));ctx.fill();if(node.id===data.champion_id){ctx.beginPath();ctx.arc(q.x,q.y,6,0,Math.PI*2);ctx.strokeStyle='#fff4c2';ctx.stroke()}});
    ctx.fillStyle=COLORS.muted;ctx.font='10px Segoe UI';ctx.textAlign='left';ctx.fillText(`generation ${minGen}`,p.l,h-8);ctx.textAlign='right';ctx.fillText(`generation ${maxGen} · champion`,w-p.r,h-8);ctx.textAlign='left';ctx.fillText(data.message||'',p.l,12);
  }
  window.BrainPanels = { drawNetwork, drawWeightMatrices, drawBars, drawDecision, drawPitch, drawTrajectory, drawSensitivity, drawPopulation, drawStats, drawFamily, COLORS };
})();
