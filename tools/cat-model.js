/*
 * 貓咪 3D 模型 — 「球體 → 二次塑型」。
 * ------------------------------------------------------------------
 * 每個部位都從實心球/圓錐/管出發（有真正的體積與受光），
 * 再對頂點做一次變形(deform)雕成有辨識度的形狀：梨形身體、寬頰圓臉…
 * 變形後重算法線 → toon 著色仍是立體的，不是平面片、也不是純圓球。
 * 花色用「頂點顏色」直接畫在體積上（白底 + 橘斑），描邊用反殼。
 *
 * 這支不綁定 three 來源（THREE 由呼叫端傳入），所以瀏覽器與 Node 檢視器可共用同一份模型。
 */

/*
 * ── 外觀（skin）──
 * 一隻貓、四種花色。每個 skin 只描述「顏色」與「花紋怎麼畫」，幾何形狀完全共用。
 *   col   : 調色盤。cream/orange/orangeD 是「淺色區/主色區/深色層次」三階，
 *           其餘 pink(耳內)/nose/out(描邊)/eye 各自獨立。dark 只有需要第三種毛色的花色才用（三花）。
 *   parts : 純色部位取哪個調色盤鍵。A = 角色左側(s<0)、B = 右側(s>0)。
 *   bodyPaint / headPaint : 頂點花色函式 (v 局部座標, c 輸出色, P 已轉成 THREE.Color 的調色盤)。
 *           省略則用預設的「斜分兩色」（＝橘白貓的畫法）。
 */
const splitBody = (v, c, P) => c.copy(0.7 * v.x - 0.2 * v.z > 0.14 ? P.orange : P.cream); // 右/後主色、前淺色
const splitHead = (v, c, P) => c.copy(0.42 * v.x + v.y > 0.05 ? P.orange : P.cream);      // 上/右主色、左下淺色
// smoothstep：x 在 [a,b] 間平滑 0→1，用於「模糊過渡」的雙色漸層（頂點花色用 c.lerp 混色）
const smooth = (x, a, b) => { let t = (x - a) / (b - a); t = t < 0 ? 0 : t > 1 ? 1 : t; return t * t * (3 - 2 * t); };

// 斑塊：以 (cx,cy,cz) 為心的橢球，邊緣用低頻正弦揉皺 → 圓角但不規則，像真的毛色斑，而不是幾何切面。
// wob 越大邊緣越破碎；0 就是乾淨的橢球。
const blob = (v, cx, cy, cz, rx, ry, rz, wob = 0.14) => {
  const x = (v.x - cx) / rx, y = (v.y - cy) / ry, z = (v.z - cz) / rz;
  const d = Math.sqrt(x * x + y * y + z * z);
  const w = wob * (Math.sin(v.x * 5.1 + v.y * 3.3) * 0.6 + Math.sin(v.z * 4.4 - v.y * 5.7) * 0.4);
  return d < 1 + w;
};

export const SKINS = {
  orangin: {
    label: 'orangin',
    col: { cream: 0xf7efe4, orange: 0xef9f57, orangeD: 0xd9822f, pink: 0xf3b3bd, nose: 0xcf7a63, out: 0x2b2320, eye: 0x241d1a },
    parts: { earA: 'cream', earB: 'orange', thighA: 'cream', thighB: 'orange', paw: 'cream', tail: 'orange', tailRing: 'orangeD', tailTip: 'orange' },
  },
  // 虎斑：暖褐底、奶油色胸腹，深褐條紋。條紋必須「成組平行、只長在背側」才像虎斑 ——
  // 灑滿全身的高頻雜紋只會糊成一團髒色，所以身體只在背上做幾道環帶、臉上只做額前直紋。
  tabby: {
    label: 'tabby',
    col: { cream: 0xe4d5b8, orange: 0xb08a5e, orangeD: 0x6b5236, pink: 0xdda6a6, nose: 0xa2705c, out: 0x2b2320, eye: 0x241d1a },
    parts: { earA: 'orange', earB: 'orange', thighA: 'orange', thighB: 'orange', paw: 'cream', tail: 'orange', tailRing: 'orangeD', tailTip: 'orangeD' },
    bodyPaint: (v, c, P) => {
      // 底色：背側褐、胸腹奶油。左右對稱（不含 x）→ 虎斑是「上深下淺」的野生保護色，不是左右分色。
      const spine = 0.55 * v.y - 0.45 * v.z;                 // 往背脊方向遞增（身體已前傾 35°）
      c.copy(spine > 0.1 ? P.orange : P.cream);
      // 環帶：沿背脊等距切帶，只畫在離明暗分界夠遠的背側 → 腹側乾淨、條紋不會咬在交界上變毛邊。
      if (spine > 0.24 && Math.sin((v.y * 0.82 + v.z * 0.57) * 14) > 0.45) c.copy(P.orangeD);
    },
    headPaint: (v, c, P) => {
      // 臉：額頭與後腦褐色，口鼻／下巴／臉頰奶油色（虎斑的白口罩）
      const muzzle = v.y < 0.02 && v.z > 0.25;
      c.copy(muzzle ? P.cream : P.orange);
      // 額前直紋：眼睛以上、臉的正面，左右對稱的細紋（用 |x| → 中線留空，兩側各兩道，像虎斑的 M 字紋）
      if (v.y > 0.3 && v.z > 0 && Math.sin(Math.abs(v.x) * 12 + 1) > 0.55) c.copy(P.orangeD);
    },
  },
  // 三花：白底 + 橘斑 + 黑斑（dark）。斑塊用 blob（不規則橢球）灑在不對稱的位置 —— 三花的重點是
  // 「左右完全不對稱的色塊」，用平面切分會變成左右對半的小丑臉，一定要用一塊一塊的斑。
  calico: {
    label: 'calico',
    col: { cream: 0xf8f3ec, orange: 0xef9f57, orangeD: 0xd9822f, dark: 0x3a322f, pink: 0xf3b3bd, nose: 0xd9908c, out: 0x2b2320, eye: 0x241d1a },
    parts: { earA: 'dark', earB: 'orange', thighA: 'cream', thighB: 'orange', paw: 'cream', tail: 'orange', tailRing: 'dark', tailTip: 'cream' },
    bodyPaint: (v, c, P) => {
      c.copy(P.cream);                                              // 白底（胸腹、四肢一律白）
      if (blob(v, 0.85, 0.35, -0.15, 0.75, 0.8, 0.95)) c.copy(P.orange);   // 右肩橘斑
      if (blob(v, 0.15, 0.7, -0.75, 0.7, 0.55, 0.6)) c.copy(P.orange);     // 背中橘斑
      if (blob(v, -0.8, 0.25, -0.35, 0.7, 0.75, 0.85)) c.copy(P.dark);     // 左背黑斑
      if (blob(v, -0.35, -0.55, 0.75, 0.42, 0.4, 0.4)) c.copy(P.dark);     // 左前腹一小塊黑
    },
    headPaint: (v, c, P) => {
      c.copy(P.cream);                                              // 白臉
      if (blob(v, 0.55, 0.6, -0.1, 0.7, 0.75, 1.0)) c.copy(P.orange);      // 右額橘斑（連到右耳）
      if (blob(v, -0.62, 0.7, -0.2, 0.6, 0.65, 0.95)) c.copy(P.dark);      // 左額黑斑（連到左耳）
      if (blob(v, -0.2, 0.95, 0.35, 0.45, 0.4, 0.5)) c.copy(P.dark);       // 頭頂偏左再一小塊 → 破對稱
    },
  },
};
export const DEFAULT_SKIN = 'orangin';

export function buildCat(THREE, opts = {}) {
  const gm = opts.gradientMap || null;
  const skin = SKINS[opts.skin] || SKINS[DEFAULT_SKIN];
  const COL = skin.col;
  const PT = skin.parts;
  const P = {};                                    // 調色盤的 THREE.Color 版（頂點花色用）
  for (const k in COL) P[k] = new THREE.Color(COL[k]);
  const pc = (key) => COL[key] ?? COL.cream;       // 部位色：parts 指到的調色盤鍵
  const toon = (c) => new THREE.MeshToonMaterial({ color: c, gradientMap: gm });
  const toonVC = () => new THREE.MeshToonMaterial({ vertexColors: true, gradientMap: gm });
  const basic = (c) => new THREE.MeshBasicMaterial({ color: c });

  const V = new THREE.Vector3();
  const deform = (geo, fn) => {
    const p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) { V.fromBufferAttribute(p, i); fn(V, i); p.setXYZ(i, V.x, V.y, V.z); }
    p.needsUpdate = true; geo.computeVertexNormals();
  };
  // 水平切出平底（底邊）：低於 cy 的頂點壓到 cy；rr>0 時在 cy 上方 rr 範圍內用 smoothstep
  // 讓斜率平滑歸零，底面與側面之間是圓弧過渡，不是硬折角。
  const flatten = (geo, cy, rr = 0) => {
    const p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const y = p.getY(i);
      if (y < cy) { p.setY(i, cy); continue; }
      if (rr > 0 && y < cy + rr) {
        const t = (y - cy) / rr;
        const e = t * t * (3 - 2 * t); // smoothstep：兩端切線斜率為 0 → 銜接處無硬邊
        p.setY(i, cy + e * (y - cy));
      }
    }
    p.needsUpdate = true; geo.computeVertexNormals();
  };
  const paint = (geo, fn) => {
    const p = geo.attributes.position, arr = [], c = new THREE.Color(), v = new THREE.Vector3();
    for (let i = 0; i < p.count; i++) { v.fromBufferAttribute(p, i); fn(v, c); arr.push(c.r, c.g, c.b); }
    geo.setAttribute('color', new THREE.Float32BufferAttribute(arr, 3));
  };
  // 依「弧位置」上色（管狀幾何）：oArr[i] = 該頂點沿尾巴的 0..1 位置 → 尾巴分段染色（如後半段變黑）。
  const paintArc = (geo, oArr, keyOf) => {
    const arr = [], c = new THREE.Color();
    for (let i = 0; i < oArr.length; i++) { c.copy(P[keyOf(oArr[i])] || P.cream); arr.push(c.r, c.g, c.b); }
    geo.setAttribute('color', new THREE.Float32BufferAttribute(arr, 3));
  };
  // 反殼粗描邊：整體縮放。只適合形狀「大致以自身原點為中心」的物件（球/橢圓體等）——
  // 對偏心、細長的形狀（例如從原點延伸出去的尾巴管），縮放中心離大半形狀很遠，
  // 描邊厚度會隨位置忽厚忽薄、甚至在某些角度露出破洞。
  const outline = (mesh, k = 0.06) => {
    const o = new THREE.Mesh(mesh.geometry, basic(COL.out));
    o.material.side = THREE.BackSide; o.scale.multiplyScalar(1 + k); mesh.add(o); return mesh;
  };
  // 反殼描邊（殼層版）：沿每個頂點自己的法線方向外推固定厚度，而非整體縮放。
  // 不管形狀離原點多遠、多細長，包覆厚度都一致 → 描邊在任何角度都完整包住整個形狀。
  // 用於偏心/細長物件（尾巴管、後續類似形狀）。
  const outlineShell = (mesh, thickness = 0.035) => {
    const src = mesh.geometry;
    const geo = src.clone();
    const pos = geo.attributes.position, nor = src.attributes.normal;
    for (let i = 0; i < pos.count; i++) {
      pos.setXYZ(i, pos.getX(i) + nor.getX(i) * thickness, pos.getY(i) + nor.getY(i) * thickness, pos.getZ(i) + nor.getZ(i) * thickness);
    }
    pos.needsUpdate = true;
    const shell = new THREE.Mesh(geo, basic(COL.out));
    shell.material.side = THREE.BackSide;
    mesh.add(shell);
    return mesh;
  };
  const bodyPaint = skin.bodyPaint || splitBody;
  const headPaint = skin.headPaint || splitHead;

  // 「可彎曲」幾何體登記表：swayUpdate() 每幀依 outerness（沿 axis 從 range[0]→range[1] 的 0..1）
  // 繞 pivot 旋轉頂點做「慣性拖尾」——身體轉動時基部跟得快、末端拖得慢；停下時彈簧會過衝再衰減，
  // 完全靜置後 lag 歸零 → 徹底靜止不再晃。offset：幾何體局部原點在擺動空間中的位置（尾端圓頭球心偏在管尾）。
  // src：拖尾要跟哪個驅動——'body'（整隻的 yaw/pitch）或 'head'（頭部朝向）。
  const flex = [];
  // 拖尾彈簧「鏈」：每個驅動軸維護 SEG+1 個節點（0=基部緊貼身體，SEG=末端）。
  // 每節被前一節（更靠身體那節）用彈簧拉著 → 延遲逐節累積：基部最快跟上/停住、末端最慢 → 自然的慣性感。
  const mkChain = () => ({ a: new Float64Array(SEG + 1), v: new Float64Array(SEG + 1) });
  const sway = { by: mkChain(), bp: mkChain(), hy: mkChain(), hp: mkChain() };
  const registerFlex = (geo, axis, range, pivot, offset, src, bend, lit = true, oArr = null) => {
    const p = geo.attributes.position;
    const base = new Float32Array(p.array);            // 原始頂點（rest 狀態，只讀）
    let o = oArr;                                       // 每頂點固定的 outerness 0..1
    if (!o) {                                           // 未指定 → 由單一座標軸推算（適用截面沿該軸為常數的盒狀鬍鬚）
      const ai = axis === 'x' ? 0 : axis === 'y' ? 1 : 2;
      const oa = axis === 'x' ? offset.x : axis === 'y' ? offset.y : offset.z;
      o = new Float32Array(p.count);
      const span = range[1] - range[0];
      for (let i = 0; i < p.count; i++) o[i] = Math.min(1, Math.max(0, (base[i * 3 + ai] + oa - range[0]) / span));
    }
    flex.push({ geo, base, o, pivot, offset, src, bend, lit });
  };
  // 管狀幾何體的 outerness：用「弧位置（第幾個截面）」而非 y 座標 → 同一截面所有頂點同一個 o，
  // 彎曲時整個截面剛體旋轉、半徑不變 → 尾巴粗細固定，不會被慣性拉扁。
  // TubeGeometry 頂點排列為 (tubular+1) 圈，每圈 (radial+1) 個 → 圈序號 = floor(i/(radial+1))。
  const tubeArcO = (geo, tubular, radial, a = 0, b = 1) => {
    const per = radial + 1, n = geo.attributes.position.count, arr = new Float32Array(n);
    for (let i = 0; i < n; i++) arr[i] = a + (b - a) * (Math.min(tubular, (i / per) | 0) / tubular);
    return arr;
  };

  const root = new THREE.Group();
  // torso：身體/頭/尾（會隨檢查模式的滑鼠「傾斜」）；legs：四肢（只跟著 root 偏轉 yaw，不隨 torso 傾斜，
  // 高度與傾斜角度永遠不變）——兩者是 root 的兄弟節點，讓 pet3d.js 能分開套用 yaw / pitch。
  const torsoG = new THREE.Group();
  const legsG = new THREE.Group();
  root.add(torsoG, legsG);

  // ---- 尾巴（捲管 + 深色環 + 圓尖）----
  const tailG = new THREE.Group();
  {
    const curve = new THREE.CatmullRomCurve3([[0, 0, 0], [.4, .5, 0], [.85, 1.05, 0], [1.0, 1.7, 0], [.7, 2.2, 0], [.15, 2.4, 0]].map((a) => new THREE.Vector3(...a)));
    // 整條尾巴共用一組彎曲參數：沿 tailG 的 y（基部 0 → 尖端 2.4）繞尾根 (0,0,0) 拖尾
    const TRANGE = [0, 2.4], TPIVOT = { x: 0, y: 0, z: 0 }, ZERO = { x: 0, y: 0, z: 0 };
    const tpaint = skin.tailPaint;                     // 有則依弧位置分段染色（頂點色），否則整條單色
    const tubeGeo = new THREE.TubeGeometry(curve, 30, 0.2, 14, false);
    const tubeO = tubeArcO(tubeGeo, 30, 14);           // 弧位置 outerness → 截面剛體旋轉、粗細不變
    if (tpaint) paintArc(tubeGeo, tubeO, tpaint);
    const tube = new THREE.Mesh(tubeGeo, tpaint ? toonVC() : toon(pc(PT.tail)));
    outlineShell(tube, 0.03); // 沿管面法線外推固定厚度 → 黑邊沿整條尾巴均勻包覆，不受偏心/角度影響
    registerFlex(tube.geometry, 'y', TRANGE, TPIVOT, ZERO, 'body', tailBend, true, tubeO);
    registerFlex(tube.children[0].geometry, 'y', TRANGE, TPIVOT, ZERO, 'body', tailBend, false, tubeO); // 黑邊殼同步彎
    tailG.add(tube);
    for (const [a, b] of [[.32, .44], [.55, .67], [.78, .9]]) {
      const pts = []; for (let i = 0; i <= 8; i++) pts.push(curve.getPoint(a + (b - a) * i / 8));
      const ringGeo = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 10, 0.205, 14, false);
      const ringO = tubeArcO(ringGeo, 10, 14, a, b);   // 對映整條尾巴的弧位置
      if (tpaint) paintArc(ringGeo, ringO, tpaint);    // 環帶也依弧位置染色 → 後半段的環一起變黑
      const ring = new THREE.Mesh(ringGeo, tpaint ? toonVC() : toon(pc(PT.tailRing)));
      registerFlex(ring.geometry, 'y', TRANGE, TPIVOT, ZERO, 'body', tailBend, true, ringO);
      tailG.add(ring);
    }
    const tip = new THREE.Mesh(new THREE.SphereGeometry(0.2, 18, 14), toon(pc(PT.tailTip)));
    outlineShell(tip, 0.03); // 尖端改用殼層黑邊 → 能與球體頂點一起彎（縮放式黑邊無法跟著變形）
    tip.position.copy(curve.getPoint(1)); tailG.add(tip);
    const tOff = { x: tip.position.x, y: tip.position.y, z: tip.position.z }; // 球心偏在管尾
    const tipO = new Float32Array(tip.geometry.attributes.position.count).fill(1); // 整顆尖端 o=1 → 剛體旋轉、不變形
    registerFlex(tip.geometry, 'y', TRANGE, TPIVOT, tOff, 'body', tailBend, true, tipO);
    registerFlex(tip.children[0].geometry, 'y', TRANGE, TPIVOT, tOff, 'body', tailBend, false, tipO);
  }
  tailG.position.set(0, -0.85, -0.7); // 接在身體正後方中央（原本偏在右側 x=0.8）
  tailG.rotation.z = -0.15;
  torsoG.add(tailG);

  // ---- 身體（橢圓體、35° 前傾、平底；更胖更短）----
  // bodyG 的原點放在身體「底部」（平切面，貼近四肢站立處）而非球心 → 抓耳等動作讓身體傾斜時，
  // 是繞底部轉（像站著的動物側身傾），不是繞肚子中心打轉。bodyMesh 把身體網格（與掛在身體上方的頭，
  // 見下方 headG）往上位移抵銷這個位移，讓靜止姿勢與原本（樞紐在球心時）完全一致。
  const BODY_CUT_Y = -0.72; // 身體平底切面（局部座標）＝新的底部樞紐基準
  const bodyG = new THREE.Group();
  const bodyMesh = new THREE.Group();
  bodyMesh.position.set(0, -BODY_CUT_Y, 0);
  bodyG.add(bodyMesh);
  const BODY_TILT = THREE.MathUtils.degToRad(35);
  /* ── 卵形收窄 ──
   * 球體 → 蛋：沿長軸把「垂直於長軸的整個截面」等比縮小，前端縮、後端不縮。
   * 縮的是整整一圈（x 與 z 同時），不是單一方向 —— 只縮 x 是把身體壓扁，
   * 只縮 z 是把身體縮短，兩者都不是蛋。
   *
   * 長軸是橢球自身的 y：前傾 35° 之後它指向「前上方」，而頭正是接在那裡
   * （HEAD_ON_BODY 在 bodyMesh 空間是 +y +z）。所以往 +y 收＝收胸廓，
   * 而髖部（-y，前傾後落在後下方）維持原寬。
   */
  // 最前端的截面縮多少（0 = 不收，0.16 = 窄 16%）。
  const BODY_FRONT_NARROW = 0.06;
  // 從長軸的哪裡開始收（-1 臀部、0 最寬的那一圈、+1 胸口最前）。
  // 設在最寬處**之前**，收窄才會真的切過那一圈；設 0 以上等於只削前端一點。
  const BODY_NARROW_FROM = -0.95;
  {
    const geo = new THREE.SphereGeometry(1.0, 52, 44);
    const rx = 1.12, ry = 1.06, rz = 1.14;                // 更胖(x,z)、更短(y 長軸)
    const ct = Math.cos(BODY_TILT), st = Math.sin(BODY_TILT);
    deform(geo, (v) => {
      // smoothstep 兩端斜率為 0：收窄平滑地長出來，不會在腰上生出新的折角。
      const narrow = 1 - BODY_FRONT_NARROW * smooth(v.y, BODY_NARROW_FROM, 1);

      // 1) 橢圓 + 卵形收窄：長軸 y 不動，截面 (x, z) 整圈同比例縮
      let x = v.x * rx * narrow, y = v.y * ry, z = v.z * rz * narrow;
      const y2 = y * ct - z * st, z2 = y * st + z * ct;    // 2) 繞 X 前傾 35°（頂端往 +z）
      y = y2; z = z2;
      if (y < BODY_CUT_Y) y = BODY_CUT_Y;                  // 3) 水平切出平底（法線朝正下方）
      v.set(x, y, z);
    });
    paint(geo, (v, c) => bodyPaint(v, c, P));
    const body = outline(new THREE.Mesh(geo, toonVC()), 0.05);
    bodyMesh.add(body);
  }
  bodyG.position.set(0, -0.5 + BODY_CUT_Y, -0.05);
  torsoG.add(bodyG);

  // ---- 後腿（大腿掛在「髖」樞紐、腳掌掛在「膝」樞紐 → 兩段關節）----
  // 抓耳時：整條後腿繞髖往上抬（大腿也抬起），腳掌再繞膝往前上伸到耳後；另外三腿不動。
  const hindHips = {}, hindPaws = {};
  for (const s of [-1, 1]) {
    // 髖關節樞紐（大腿頂後方）→ 整條後腿繞此上抬
    const hip = new THREE.Group();
    hip.position.set(s * 0.82, -0.45, -0.45);
    // 大腿 / 後臀（掛在髖上；與身體之間有黑邊）
    const tg = new THREE.SphereGeometry(0.46, 26, 20);
    deform(tg, (v) => { v.z *= 1.12; });                // 略前後拉長
    const thigh = outline(new THREE.Mesh(tg, toon(pc(s < 0 ? PT.thighA : PT.thighB))), 0.07);
    thigh.position.set(0, -0.35, 0.15);                 // 淨世界位置 ≈ (s*0.92, -0.8, -0.25)
    hip.add(thigh);
    // 膝關節樞紐（掛在髖上）→ 腳掌繞膝再往上前伸
    const knee = new THREE.Group();
    knee.position.set(0, -0.4, 0.25);                   // 淨世界位置 ≈ (s*0.92, -0.85, -0.15)
    // 腳掌（畫在大腿之後 → 疊在前方；重疊處只留一條乾淨黑邊 = 大小腿分界；底邊切平）
    const pg = new THREE.SphereGeometry(0.24, 22, 18);
    deform(pg, (v) => { v.y *= 0.8; v.z *= 1.45; });    // 壓扁往前拉長
    flatten(pg, -0.26, -0.3);                             // 腳掌底邊，圓弧過渡
    const hpaw = outline(new THREE.Mesh(pg, toon(pc(PT.paw))), 0.08);
    hpaw.position.set(s * 0.14, -0.47, 0.21);           // 淨世界位置 ≈ (s*1.06, -1.32, 0.06)（前下方、略比前腳寬）
    knee.add(hpaw); hip.add(knee); legsG.add(hip);
    hindHips[s < 0 ? 'R' : 'L'] = hip;
    hindPaws[s < 0 ? 'R' : 'L'] = knee;
  }

  // ---- 前腳掌（前傾 30°，但底面維持水平、不隨腳掌傾斜）----
  // 前腳掌掛在「肩關節」樞紐上 → 舔前腳動作時繞肩往上抬到嘴邊。
  const frontPaws = {};
  const PAW_TILT = THREE.MathUtils.degToRad(75);
  for (const s of [-0.8, 0.8]) {
    const geo = new THREE.SphereGeometry(0.26, 20, 16);
    const ct = Math.cos(PAW_TILT), st = Math.sin(PAW_TILT);
    deform(geo, (v) => {
      let y = v.y * 0.7, z = v.z * 1.25;                  // 1) 壓扁塑形
      const y2 = y * ct - z * st, z2 = y * st + z * ct;   // 2) 繞 X 前傾 30°
      v.y = y2; v.z = z2;
    });
    flatten(geo, -0.26, -0.3);                              // 3) 前傾「之後」再水平切，圓弧過渡 → 底面平面、不傾斜
    const paw = outline(new THREE.Mesh(geo, toon(pc(PT.paw))), 0.09);
    const sh = new THREE.Group();
    sh.position.set(s * 0.44, -0.72, 0.5);               // 肩關節樞紐
    paw.position.set(0, -0.56, 0.26);                    // 淨世界位置 ≈ (s*0.44, -1.28, 0.76)
    sh.add(paw); legsG.add(sh);
    frontPaws[s < 0 ? 'L' : 'R'] = sh;
  }

  // ---- 頭（大圓球 → 飽滿圓臉 + 前突口鼻；依紅色弧線的球面走向）----
  // headG 掛在 bodyMesh（身體頂端）上、而非用相對 torsoG 的絕對座標定位 → 身體傾斜/移動時，
  // 頭是「長在身體頂端」跟著一起動，不是各自獨立算絕對位置。headG 的原點放在頭部「底部」
  // （球體最低點，貼近頸部）而非球心 → 點頭/搖頭/傾斜都繞底部轉，貼近真實脖子轉動的感覺。
  // headMesh 把所有頭部子物件往上位移 HEAD_R 抵銷這個位移，讓靜止姿勢與原本（樞紐在球心時）完全一致。
  const HEAD_R = 1.12;
  const HEAD_ON_BODY = { x: 0, y: 1.05 - (-0.5), z: 0.48 - (-0.05) }; // 頭中心相對「身體頂端」bodyMesh 的偏移
  const headG = new THREE.Group();
  headG.position.set(HEAD_ON_BODY.x, HEAD_ON_BODY.y - HEAD_R, HEAD_ON_BODY.z); // 頭部底部（= 頭中心往下一個半徑）
  const headMesh = new THREE.Group();
  headMesh.position.set(0, HEAD_R, 0); // 把頭部視覺內容移回原本（頭中心）的位置
  headG.add(headMesh);
  {
    const R = HEAD_R;
    // 頭部細分：預設 52×40。頂點花色是逐三角形線性內插，臉上有斜邊色帶的花色（如 red）可用 skin.headSeg
    // 提高細分 → facet 越細＝色帶邊緣越平滑、越不鋸齒；其他花色保留原本細分。
    const [hw, hh] = skin.headSeg || [52, 40];
    const geo = new THREE.SphereGeometry(R, hw, hh);
    deform(geo, (v) => {
      const ny = v.y / R, nz = v.z / R;
      const front = Math.max(0, nz);
      const t = Math.min(1, Math.max(0, (ny + 1) / 2));
      v.x *= THREE.MathUtils.lerp(1.06, 0.94, t);             // 大致圓，僅一點下寬上窄
      const cheek = Math.exp(-(((ny + 0.15) / 0.55) ** 2));   // 兩頰飽滿（往外凸）
      v.x *= 1 + 0.09 * cheek;
      // 三角形口鼻平台：頂點(鼻)在上、往下加寬到下巴；頂角(鼻)稍微前傾 → 鼻端最凸
      const apexY = -0.06, baseY = -0.62, halfW = 0.46;
      if (front > 0 && ny < apexY && ny > baseY) {
        const down = (apexY - ny) / (apexY - baseY);          // 0 頂(鼻) → 1 底(下巴)
        const wAt = 0.05 + halfW * down;                      // 三角寬度
        const e = 1 - Math.min(1, Math.abs(v.x) / wAt);       // 1 中線 → 0 邊緣
        if (e > 0) {
          const p = Math.min(1, e / 0.5); const plateau = p * p * (3 - 2 * p); // 平台：中間平、邊緣收
          v.z += (0.05 + 0.14 * (1 - down)) * plateau * front; // 頂角前傾：越靠鼻越往前
        }
      }
    });
    paint(geo, (v, c) => headPaint(v, c, P));
    const head = outline(new THREE.Mesh(geo, toonVC()), 0.05);
    headMesh.add(head);
  }

  // 耳（圓錐，尖端收圓）— 一橘一白
  // 耳：球體→二次塑型的圓弧尖角（高*1.2、寬*1.5）；縱軸再往外側旋轉 15°
  const ears = {};
  const EAR_YAW = THREE.MathUtils.degToRad(15);
  const EAR_LEAN = THREE.MathUtils.degToRad(15);
  for (const s of [-1, 1]) {
    const earG = new THREE.Group();
    earG.rotation.order = 'YXZ';                                 // 先 yaw 定方向，再在該方向上傾斜
    earG.position.set(s * 0.62, 0.88, -0.05);
    earG.rotation.z = -s * 0.28; earG.userData.base = -s * 0.28; // 外傾（roll，applyPose 會重設）
    earG.rotation.y = s * EAR_YAW;                               // 縱軸往外側偏轉 15°
    earG.rotation.x = EAR_LEAN;                                  // 朝偏轉後的方向再傾斜 15°
    const R = 0.6;                                               // 底半寬 0.6 → 全寬 1.2（≈ 原 0.8 * 1.5）
    const geo = new THREE.SphereGeometry(R, 30, 24);
    deform(geo, (v) => {
      const t = (v.y / R + 1) / 2;                     // 0 底 1 頂
      const w = THREE.MathUtils.lerp(1.0, 0.16, t);    // 底寬頂窄；頂端沿球面收 → 圓弧尖角
      v.x *= w; v.z *= w;
      v.y *= 0.82;                                     // 高度 2R*0.82 ≈ 0.98（≈ 原 0.82 * 1.2）
    });
    const outer = outline(new THREE.Mesh(geo, toon(pc(s < 0 ? PT.earA : PT.earB))), 0.09);
    earG.add(outer);
    const ig = new THREE.SphereGeometry(0.36, 22, 18);
    deform(ig, (v) => { const t = (v.y / 0.36 + 1) / 2; const w = THREE.MathUtils.lerp(1.0, 0.2, t); v.x *= w; v.z *= w; v.y *= 0.8; });
    const inner = new THREE.Mesh(ig, toon(COL.pink)); inner.position.set(0, -0.04, 0.24); earG.add(inner);
    headMesh.add(earG); ears[s < 0 ? 'L' : 'R'] = earG;
  }

  // 眼（黑豆眼 + 高光）— eyeAngry 皮膚保留原本眼形，再用一個傾斜平面「斜切」上緣：
  // 削掉的頂點壓到切面上 → 上緣是一條硬邊斜線（外角高、內角低），是真的切面而非變形。
  const eyes = [];
  const pupils = [];                                          // 黑瞳（僅 eyeTwoLayer）；applyPose 每幀反向偏移 → 懸浮凝視感
  const eyeAngry = skin.eyeAngry || 0;
  // 黑白兩層同心圓眼：壓扁成圓盤的白眼球 + 前方一顆黑瞳（正面看＝兩層圓圈）。眨眼一樣靠 scale.y。
  if (skin.eyeTwoLayer) for (const s of [-1, 1]) {
    // 明顯拉長的杏仁形白眼球（長軸夠長，上挑角度才讀得出來）+ 黑瞳；繞 z 讓眼尾(外角)高、眼頭(內角)低
    const wg = new THREE.SphereGeometry(0.27, 28, 20);
    deform(wg, (v) => { v.x *= 1.15; v.y *= 0.68; v.z *= 0.4; }); // 壓扁朝前 + 左右拉長、上下收窄 → 清楚的橫向杏仁
    const white = new THREE.Mesh(wg, basic(COL.eyeW ?? 0xffffff));
    const bg = new THREE.SphereGeometry(0.14, 22, 18);
    deform(bg, (v) => { v.x *= 1.05; v.z *= 0.4; });
    const black = new THREE.Mesh(bg, basic(COL.eye));
    black.position.set(s * 0.1, 0.06, 0.08);                     // 疊在白眼球正前方
    black.userData.base = { x: s * 0.1, y: 0, z: 0.08 };      // 靜止基準；applyPose 在此之上加反向偏移
    black.userData.tilt = s * 0.3;                            // 白眼球的 z 上挑角 → 偏移要先扣掉它，左右才對稱
    white.add(black); pupils.push(black);
    white.position.set(s * 0.37, 0.12, 0.95);
    white.rotation.z = s * 0.3;                                // 外角上挑（眼尾高、眼頭低）
    headMesh.add(white); eyes.push(white);
  }
  else for (const s of [-1, 1]) {
    const geo = new THREE.SphereGeometry(0.17, 18, 14);
    deform(geo, (v) => { v.y *= 1.25; v.z *= 0.55; });          // 原本的直立豆眼
    if (eyeAngry) deform(geo, (v) => {
      const cut = 0.02 + eyeAngry * (s * v.x);                  // 切面：外側(s*x>0)高、內側低
      if (v.y > cut) v.y = cut;                                 // 上緣以上一律壓到斜切面
    });
    const eye = new THREE.Mesh(geo, basic(COL.eye));
    eye.position.set(s * 0.42, 0.12, 0.98);
    // eyeAngry：高光改亮紅（非白）並稍微下移，避免壓在斜切上緣、也更配紅色凶眼
    const shine = new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 8), basic(eyeAngry ? 0xff6b5a : 0xffffff));
    const sp = { x: eyeAngry ? 0 : -0.06, y: eyeAngry ? 0.0 : 0.08, z: eyeAngry ? 0.1 : 0.14 };
    shine.position.set(sp.x, sp.y, sp.z);
    // 高光也用同一個斜切面削平（換算到 shine 局部座標）→ 白點上緣與紅眼上緣同斜率，不會凸出斜切邊
    if (eyeAngry) deform(shine.geometry, (v) => {
      const cut = 0.02 + eyeAngry * (s * (sp.x + v.x)) - sp.y;
      if (v.y > cut) v.y = cut;
    });
    eye.add(shine);
    headMesh.add(eye); eyes.push(eye);
  }
  // 鼻
  { const geo = new THREE.SphereGeometry(0.12, 16, 12); deform(geo, (v) => { v.x *= 1.3; v.y *= 0.8; v.z *= 0.7; });
    const nose = new THREE.Mesh(geo, toon(COL.nose)); nose.position.set(0, -0.1, 1.26); headMesh.add(nose); }
  // 嘴：一般是微笑弧（‿）；flatMouth 一條水平直線；mouthW 兩個 U 相接（‿‿，有弧度的 w）
  if (skin.mouthW) {
    const R = 0.085;                                          // 每個 U 的半徑；兩 U 圓心各在 ±R → 內端在中線相接成 w
    for (const s of [-1, 1]) {
      const u = new THREE.Mesh(new THREE.TorusGeometry(R, 0.02, 6, 16, Math.PI), basic(COL.out));
      u.position.set(s * R, -0.32, 1.2); u.rotation.z = Math.PI; // 轉半圈 → 開口朝上（微笑 U）
      headMesh.add(u);
    }
    // 中間一條垂直線：從鼻底連到兩 U 相接處（人中／貓的口鼻縫）
    const philtrum = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.24, 0.03), basic(COL.out));
    philtrum.position.set(0, -0.2, 1.22); headMesh.add(philtrum);
  } else if (skin.flatMouth) {
    const mouth = new THREE.Mesh(new THREE.BoxGeometry(0.26, 0.03, 0.03), basic(COL.out));
    mouth.position.set(0, -0.38, 1.18); headMesh.add(mouth);
  } else {
    const mouth = new THREE.Mesh(new THREE.TorusGeometry(0.12, 0.02, 6, 16, Math.PI), basic(COL.out));
    mouth.position.set(0, -0.32, 1.2); mouth.rotation.z = Math.PI; headMesh.add(mouth);
  }
  // 尖牙：嘴角兩側朝正上方的三角尖峰（^）— 直立不傾斜，與平嘴的 - 組成 ^-^ 的凶臉嘴形。
  // 圓錐正面剪影＝三角形＝^；底寬 0.05、對齊平嘴直線兩端(±0.13)、底貼在嘴線上、尖端朝上。
  if (skin.fangs) for (const s of [-1, 1]) {
    const fang = new THREE.Mesh(new THREE.ConeGeometry(0.05, 0.14, 8), basic(COL.out));
    fang.position.set(s * 0.18, -0.33, 1.18); headMesh.add(fang);
  }
  // 鬍鬚（兩側對稱；沿長度分段 → 頭轉動時鬚尖慣性拖尾、停下微微回擺）
  for (const s of [-1, 1]) for (let i = 0; i < 3; i++) {
    const wk = new THREE.Mesh(new THREE.BoxGeometry(0.66, 0.03, 0.03, 14, 1, 1), basic(COL.out));
    wk.position.set(s * 0.92, 0.0 - i * 0.14, 0.9);
    wk.rotation.z = -s * (1 - i) * 0.22;
    const range = s > 0 ? [-0.33, 0.33] : [0.33, -0.33];            // outerness：臉頰端 0 → 鬚尖 1
    // 繞鬚根（臉頰端）彎；在鬍鬚自身局部空間彎，s 讓左右兩側往同一世界方向拖尾
    registerFlex(wk.geometry, 'x', range, { x: -s * 0.33, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }, 'head',
      (lagY, lagP) => whiskerBend(lagY, lagP, s), false);
    headMesh.add(wk);
  }

  bodyMesh.add(headG); // 頭掛在身體頂端（bodyMesh），跟著身體一起移動/傾斜
  return { root, torso: torsoG, legs: legsG, head: headG, earL: ears.L, earR: ears.R, tail: tailG,
    body: bodyMesh, bodyPivot: bodyG, eyes, pupils,
    pawFL: frontPaws.L, pawFR: frontPaws.R, pawHL: hindPaws.L, pawHR: hindPaws.R,
    hipHL: hindHips.L, hipHR: hindHips.R, flex, sway };
}

// ── 慣性拖尾（彈簧鏈）── 沿長度切成 SEG 節；每節被「更靠身體」的前一節用彈簧拉著。
// 延遲逐節累積 → 末端加速最慢、減速也最慢；基部最快跟上也最快停 → 慣性感自然浮現。
// ── 慣性拖尾鏈參數 ──
const SEG = 8;                                               // 幾個節點（越多越綿密光滑、越少越硬朗）
const CH_K = 240;                                            // 剛度：越大拖尾越短、反應越快；越小越長尾感
const CH_C = 15;                                             // 阻尼：越大停下越快、越少回擺；越小越會搖晃幾下才停
function updateChain(cs, drive, d) {
  cs.a[0] = drive; cs.v[0] = 0;                                      // 基部完全跟著身體（最快，無延遲）
  for (let i = 1; i <= SEG; i++) {
    const acc = CH_K * (cs.a[i - 1] - cs.a[i]) - CH_C * cs.v[i];     // 只被前一節拉（單向）→ 前導後隨、不回拖身體
    cs.v[i] += acc * d;
    cs.a[i] += cs.v[i] * d;
  }
}
function sampleChain(cs, o) {                                        // o(0..1)→該處相對基部的落後角（負值＝往後拖）
  const x = o * SEG; let i = Math.floor(x); if (i >= SEG) i = SEG - 1;
  const f = x - i;
  return cs.a[i] * (1 - f) + cs.a[i + 1] * f - cs.a[0];
}
function chainActivity(cs) {
  let m = 0; for (let i = 1; i <= SEG; i++) m += Math.abs(cs.a[i] - cs.a[0]) + Math.abs(cs.v[i]);
  return m;
}

// ── 拖尾增益 ── 決定慣性拖尾對各朝向變化的反應大小。
// 越大 → 拖尾振幅越大、擺動越誇張；越小 → 跟著身體越緊、表現越受約束。
const TAIL_GY = 0.2;                                        // 尾巴對 yaw(左右轉) 的增益 → 平面內側擺幅
const TAIL_GP = 0.2;                                        // 尾巴對 pitch(上下仰) 的增益 → 前後飄幅
function tailBend(lagY, lagP) {
  return { ax: lagP * TAIL_GP, ay: 0, az: lagY * TAIL_GY };
}

const WK_GY = 0.05;                                         // 鬍鬚對 yaw 的增益（越大越甩）
const WK_GP = 0.05;                                         // 鬍鬚對 pitch 的增益（越大越翹）
function whiskerBend(lagY, lagP, s) {
  return { ax: 0, ay: lagY * WK_GY * s, az: lagP * WK_GP * s };  // s：±1 讓左右鬍鬚往同一世界方向拖
}

// ── 微風 ── 尾巴永遠輕輕飄（獨立於慣性拖尾）；用兩個低頻正弦疊加（陣風感），沿 o 線性增強 → 尾根不動、尾梢飄最多。
// 用弧位置 o（每截面同值）→ 截面剛體旋轉，粗細不變。修改後面的係數調整風力大小。
function windAz(o, t) {                                     // 平面內側擺（Z軸旋轉）
  return (0.05 * Math.sin(t * 1.3 + o * 2.0) + 0.03 * Math.sin(t * 0.7 + 0.5)) * o;  // 0.05 / 0.03 可調風強
}
function windAx(o, t) {                                     // 前後飄（X軸旋轉）
  return 0.035 * Math.sin(t * 1.0 + o * 1.6 + 1.0) * o;    // 0.035 可調前後風強
}

// 每幀呼叫：先推進四條拖尾鏈（body/head × yaw/pitch）追上當前朝向，再對每個頂點依 outerness 取樣落後角、
// 加上尾巴的微風，繞 pivot 旋轉。尾巴有風 → 永遠更新；鬍鬚無風 → 頭靜止時凍結、省開銷。
export function swayUpdate(parts, dt, t = 0) {
  const S = parts.sway; if (!S || !parts.flex) return;
  const d = Math.min(0.05, Math.max(0, dt));                         // 夾住 dt → 分頁切回時不會爆衝
  // 驅動角：pet3d 已在本幀設好各物件旋轉；尾巴跟整隻(body)、鬍鬚跟頭(head)
  const bY = parts.root.rotation.y, bP = parts.torso.rotation.x;
  // 尾巴 yaw 鏈同時被「身體轉向 + 尾巴自身甩動（tail.rotation.z 扣掉 -0.15 靜止偏移）」驅動 →
  // 甩動也走這條慣性鏈：基部先到、尖端逐節落後（近身快、近梢慢的原則套用到甩動，而非只有身體轉向）。
  const tailWag = parts.tail.rotation.z + 0.15;
  const hY = bY + parts.head.rotation.y, hP = bP + parts.head.rotation.x;  // 鬍鬚只跟頭，不含尾巴甩動
  updateChain(S.by, bY + tailWag, d); updateChain(S.bp, bP, d);
  updateChain(S.hy, hY, d); updateChain(S.hp, hP, d);
  const headQuiet = chainActivity(S.hy) + chainActivity(S.hp) < 2e-3; // 頭幾乎不動 → 鬍鬚可凍結
  for (const f of parts.flex) {
    const head = f.src === 'head';
    if (head && headQuiet && S.headRested) continue;                // 鬍鬚已在靜止幀寫過 → 略過
    const yc = head ? S.hy : S.by, pc = head ? S.hp : S.bp;
    const p = f.geo.attributes.position, b = f.base, os = f.o, pv = f.pivot, of = f.offset;
    for (let i = 0; i < p.count; i++) {
      const j = i * 3, oo = os[i];
      let x = b[j] + of.x - pv.x, y = b[j + 1] + of.y - pv.y, z = b[j + 2] + of.z - pv.z;
      const bd = f.bend(sampleChain(yc, oo), sampleChain(pc, oo));
      let ax = bd.ax, ay = bd.ay, az = bd.az;
      if (!head) { az += windAz(oo, t); ax += windAx(oo, t); }       // 尾巴：慣性拖尾之上再疊微風
      if (az) { const c = Math.cos(az), s = Math.sin(az), nx = x * c - y * s; y = x * s + y * c; x = nx; }
      if (ax) { const c = Math.cos(ax), s = Math.sin(ax), ny = y * c - z * s; z = y * s + z * c; y = ny; }
      if (ay) { const c = Math.cos(ay), s = Math.sin(ay), nx = x * c + z * s; z = -x * s + z * c; x = nx; }
      p.setXYZ(i, pv.x + x - of.x, pv.y + y - of.y, pv.z + z - of.z);
    }
    p.needsUpdate = true;
    if (f.lit) f.geo.computeVertexNormals();                         // 有受光的部位要重算法線；黑邊殼不用
  }
  S.headRested = headQuiet;
}

const HEAD_LEAN = 0.12; // 頭部靜止時稍微前傾（弧度）
const LIFT_HIP = 1.15;  // 抓耳：整條後腿繞髖上抬的基準角（弧度）；越大大腿平均抬得越高
const LIFT_KNEE = 1.6;  // 抓耳：腳掌繞膝再往前上伸的基準角（弧度）；越大平均越貼近耳後
const SWING_HIP = 0.4;  // 抓耳：疊加在髖上的前後擺動幅度（弧度）；越大整條腿擺得越開，越像真的在抓
const SWING_KNEE = 0.55;// 抓耳：疊加在膝上的前後擺動幅度（弧度）；與 SWING_HIP 同相位、幅度可分開調
// 黑瞳兩層行為：①偏移＝頭偏轉時黑瞳往反向「平移」（懸浮凝視感）；②偏轉＝在此基礎上，黑瞳圓盤再往反向
// 「轉」半個頭角 → 半面向正面。兩者疊加：平移把黑瞳挪到一側、偏轉讓它同時半朝鏡頭。
const PUPIL_GX = 0.16;   // 偏移：對 headYaw 的水平反向平移增益
const PUPIL_GY = 0.10;   // 偏移：對 headPitch 的垂直反向平移增益
const PUPIL_MAX = 0.13;  // 偏移上限（白眼球半徑內），避免黑瞳被推出白眼球外
const PUPIL_FRONT = 0.5; // 偏轉：補頭偏轉角的比例（0.5 = 半面向正面；1 = 完全面向正面；0 = 不轉）
// 身體傾斜量由大腦的 A.lean 直接給角度（與頭傾斜分開調 → 頭比身體再多傾一點）

export function applyPose(parts, A) {
  const scratch = A.scratch || 0, hp = Math.max(A.hpL || 0, A.hpR || 0); // 抓耳擺動相位（±1），乘上抬腳量才生效
  parts.head.rotation.set(
    HEAD_LEAN + A.headPitch * 0.8,
    A.headYaw * 0.85,
    A.headTilt + scratch * 0.05 * hp,                    // 抓耳時頭隨後腳擺動快速微顫
  );
  parts.earL.rotation.z = parts.earL.userData.base + A.earL * 0.6;
  parts.earR.rotation.z = parts.earR.userData.base - A.earR * 0.6;
  parts.tail.rotation.z = -0.15 + A.tailYaw;
  // 身體傾斜：抓耳時身體（連同掛在其頂端的頭）繞底部樞紐（bodyPivot）往該側傾；
  // 腿在 legsG、尾巴是 torsoG 的另一個兄弟節點，都不隨之傾 → 另外三腿與尾巴站定不動。
  parts.bodyPivot.rotation.z = A.lean || 0;
  // 後腿：整條繞髖上抬到基準角，再疊加 scratch 前後擺動（髖、膝同相位一起擺）→ 整條腿真的在來回抓，
  // 不只是腳掌末端小抖動。
  const hpl = A.hpL || 0, hpr = A.hpR || 0;
  if (parts.hipHL) parts.hipHL.rotation.x = -hpl * (LIFT_HIP + scratch * SWING_HIP);
  if (parts.hipHR) parts.hipHR.rotation.x = -hpr * (LIFT_HIP + scratch * SWING_HIP);
  if (parts.pawHL) parts.pawHL.rotation.x = -hpl * (LIFT_KNEE + scratch * SWING_KNEE);
  if (parts.pawHR) parts.pawHR.rotation.x = -hpr * (LIFT_KNEE + scratch * SWING_KNEE);
  const s = Math.max(0.08, A.eyeOpen);
  for (const e of parts.eyes) e.scale.y = s;
  // 黑瞳：①偏移（反向平移，夾在白眼球內）②偏轉（在偏移基礎上再往反向轉半個頭角 → 半面向正面）
  if (parts.pupils && parts.pupils.length) {
    const yaw = A.headYaw * 0.85;                    // 頭實際 yaw 角（同 parts.head.rotation.y）
    const pitch = HEAD_LEAN + A.headPitch * 0.8;     // 頭實際 pitch 角（含靜止前傾）
    const clamp = (x) => (x < -PUPIL_MAX ? -PUPIL_MAX : x > PUPIL_MAX ? PUPIL_MAX : x);
    const px = clamp(-PUPIL_GX * A.headYaw);         // 偏移：頭轉向一側 → 黑瞳往另一側
    const py = clamp(PUPIL_GY * A.headPitch);        // 偏移：頭低(正 pitch) → 黑瞳往上
    const rx = -PUPIL_FRONT * pitch, ry = -PUPIL_FRONT * yaw; // 偏轉：反向轉半個頭角 → 半朝鏡頭
    for (const b of parts.pupils) {
      const bp = b.userData.base, c = Math.cos(b.userData.tilt), sn = Math.sin(b.userData.tilt);
      // 偏移先扣掉白眼球的 z 上挑角（左右反號）→ 兩眼在螢幕上同向移動、不會一上一下
      b.position.set(bp.x + px * c + py * sn, bp.y - px * sn + py * c, bp.z);
      b.rotation.set(rx, ry, 0);
    }
  }
  parts.body.scale.set(1 - A.breathe * 0, 1 + A.breathe * 0, 1 - A.breathe * 0); // 呼吸變形暫時停用（係數歸零；恢復把 0 改回 0.02/0.03/0.02）
  parts.root.position.y = A.bob;
  parts.root.rotation.z = A.headTilt * 0.04;
  // 觸碰跟隨時身體隨頭小幅同向轉：root.y 整隻（含腿）一起偏轉，torso.x 只有身體/頭/尾傾斜
  // （腿在 legsG、是 root 的兄弟節點，不會繼承 torso 的傾斜 → 高度與傾斜角度永遠不變）。
  parts.root.rotation.y = A.bodyYaw || 0;
  parts.torso.rotation.x = A.bodyPitch || 0;
}
