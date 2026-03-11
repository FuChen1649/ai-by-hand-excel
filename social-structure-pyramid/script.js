const levels = [
  {
    index: '第 1 层',
    title: '塔尖决策层',
    description: '负责总体方向、资源配置与重大决策。',
    width: 26,
    color: '#6a5af9',
    people: [
      { label: '决策者', outfit: 'suit', color: '#2f3e8f', accessory: 'badge' }
    ]
  },
  {
    index: '第 2 层',
    title: '高层管理层',
    description: '统筹跨领域协作，将战略目标拆分为组织行动。',
    width: 34,
    color: '#5d78ff',
    people: [
      { label: '管理者', outfit: 'formal', color: '#3451c6', accessory: 'tie' }
    ]
  },
  {
    index: '第 3 层',
    title: '专业精英层',
    description: '包含科研、金融、法律、技术等高专业度岗位。',
    width: 42,
    color: '#4b8dff',
    people: [
      { label: '专家', outfit: 'lab', color: '#ffffff', accessory: 'clipboard' },
      { label: '分析师', outfit: 'formal', color: '#1f5cab', accessory: 'tablet' }
    ]
  },
  {
    index: '第 4 层',
    title: '组织骨干层',
    description: '承担管理执行、流程衔接和关键岗位支撑。',
    width: 50,
    color: '#33a1ff',
    people: [
      { label: '主管', outfit: 'vest', color: '#0f6b94', accessory: 'badge' },
      { label: '教师', outfit: 'coat', color: '#ffc857', accessory: 'book' }
    ]
  },
  {
    index: '第 5 层',
    title: '城市稳定层',
    description: '典型的白领、店长、技师等，支撑城市日常运转。',
    width: 58,
    color: '#21b7b7',
    people: [
      { label: '白领', outfit: 'shirt', color: '#4f7cff', accessory: 'tie' },
      { label: '店长', outfit: 'apron', color: '#7d4ce0', accessory: 'nameTag' },
      { label: '技师', outfit: 'overall', color: '#2c7a54', accessory: 'toolbelt' }
    ]
  },
  {
    index: '第 6 层',
    title: '基层服务层',
    description: '提供餐饮、零售、客服、配送等贴近日常生活的服务。',
    width: 66,
    color: '#36c486',
    people: [
      { label: '服务员', outfit: 'apron', color: '#f97316', accessory: 'tray' },
      { label: '客服', outfit: 'blazer', color: '#0f766e', accessory: 'headset' },
      { label: '配送员', outfit: 'jacket', color: '#2563eb', accessory: 'bag' }
    ]
  },
  {
    index: '第 7 层',
    title: '产业劳动层',
    description: '工厂、物流、维修等岗位，是生产与流通的重要力量。',
    width: 74,
    color: '#77c043',
    people: [
      { label: '工人', outfit: 'overall', color: '#355c7d', accessory: 'helmet' },
      { label: '司机', outfit: 'uniform', color: '#475569', accessory: 'cap' },
      { label: '维修员', outfit: 'jumpsuit', color: '#dc2626', accessory: 'wrench' }
    ]
  },
  {
    index: '第 8 层',
    title: '基础保障层',
    description: '农业、环卫、保洁等工作保障社会基本秩序和供应。',
    width: 82,
    color: '#e8b33f',
    people: [
      { label: '农务者', outfit: 'field', color: '#65a30d', accessory: 'strawHat' },
      { label: '环卫员', outfit: 'reflective', color: '#f59e0b', accessory: 'broom' },
      { label: '保洁员', outfit: 'uniform', color: '#0ea5e9', accessory: 'gloves' }
    ]
  },
  {
    index: '第 9 层',
    title: '广泛支撑层',
    description: '数量最庞大、覆盖最广的普通劳动与生活人群，是金字塔的基础。',
    width: 90,
    color: '#f28f5b',
    people: [
      { label: '居民', outfit: 'casual', color: '#8b5cf6', accessory: 'none' },
      { label: '青年', outfit: 'hoodie', color: '#ef4444', accessory: 'backpack' },
      { label: '家庭成员', outfit: 'casual', color: '#14b8a6', accessory: 'scarf' },
      { label: '劳动者', outfit: 'shirt', color: '#3b82f6', accessory: 'none' }
    ]
  }
];

const reverseLevels = [
  {
    index: '第 1 层',
    title: '权力顶层',
    description: '倒置金字塔的顶端，以权力符号表示对整体结构的集中影响。',
    width: 90,
    color: '#7c3aed',
    symbols: [{ label: '权力', type: 'power' }]
  },
  {
    index: '第 2 层',
    title: '高位资本层',
    description: '资源向上聚拢后的重要承接层。',
    width: 82,
    color: '#8b5cf6',
    symbols: [{ label: '金钱', type: 'money' }]
  },
  {
    index: '第 3 层',
    title: '资本枢纽层',
    description: '连接更高控制力与更广资金流转。',
    width: 74,
    color: '#a855f7',
    symbols: [{ label: '金钱', type: 'money' }]
  },
  {
    index: '第 4 层',
    title: '资本分配层',
    description: '在更大范围内进行投放、配置与吸纳。',
    width: 66,
    color: '#c026d3',
    symbols: [{ label: '金钱', type: 'money' }]
  },
  {
    index: '第 5 层',
    title: '资本运转层',
    description: '形成持续的收益循环与价值转移。',
    width: 58,
    color: '#db2777',
    symbols: [{ label: '金钱', type: 'money' }]
  },
  {
    index: '第 6 层',
    title: '交易扩散层',
    description: '让资金以更密集频率参与流通。',
    width: 50,
    color: '#ea580c',
    symbols: [{ label: '金钱', type: 'money' }]
  },
  {
    index: '第 7 层',
    title: '市场吸纳层',
    description: '接收更广泛的支付、消费与投入。',
    width: 42,
    color: '#f59e0b',
    symbols: [{ label: '金钱', type: 'money' }]
  },
  {
    index: '第 8 层',
    title: '基础汇集层',
    description: '大量零散价值在此被不断聚合。',
    width: 34,
    color: '#fbbf24',
    symbols: [{ label: '金钱', type: 'money' }]
  },
  {
    index: '第 9 层',
    title: '倒锥底部',
    description: '最尖窄处仍以金钱符号表示持续汇入。',
    width: 26,
    color: '#f59e0b',
    symbols: [{ label: '金钱', type: 'money' }]
  }
];

const legend = document.getElementById('legend');
const pairedPyramid = document.getElementById('paired-pyramid');
const personTemplate = document.getElementById('person-template');
const ruleToggle = document.getElementById('rule-toggle');
const clockTrigger = document.getElementById('clock-trigger');

const ruleModes = ['法律&法规', '公俗良约&道德'];
let ruleModeIndex = 0;

function renderLegend() {
  levels.forEach((level) => {
    const chip = document.createElement('div');
    chip.className = 'legend-chip';
    chip.innerHTML = `<span class="legend-dot" style="background:${level.color}"></span>${level.index} · ${level.title}`;
    legend.appendChild(chip);
  });

  [
    { label: '倒置金字塔 · 金钱', color: '#f59e0b' },
    { label: '倒置金字塔 · 权力', color: '#7c3aed' }
  ].forEach((item) => {
    const chip = document.createElement('div');
    chip.className = 'legend-chip';
    chip.innerHTML = `<span class="legend-dot" style="background:${item.color}"></span>${item.label}`;
    legend.appendChild(chip);
  });
}

function renderSymmetricPyramids() {
  levels.forEach((leftLevel, index) => {
    const rightLevel = reverseLevels[index];
    const row = document.createElement('div');
    row.className = 'pair-row';
    row.style.setProperty('--row-height', `${computeRowHeight(leftLevel, rightLevel)}px`);

    const leftShell = document.createElement('div');
    leftShell.className = 'level-shell level-shell-left';
    leftShell.appendChild(createPeopleLevel(leftLevel, 'left-level'));

    const spacer = document.createElement('div');
    spacer.className = 'pair-spacer';

    const rightShell = document.createElement('div');
    rightShell.className = 'level-shell level-shell-right';
    rightShell.appendChild(createSymbolLevel(rightLevel, 'right-level'));

    row.append(leftShell, spacer, rightShell);
    pairedPyramid.appendChild(row);
  });
}

function createPeopleLevel(level, sideClass) {
  const peopleMarkup = level.people
    .map((person) => {
      const node = personTemplate.content.firstElementChild.cloneNode(true);
      node.querySelector('.avatar').innerHTML = buildAvatar(person);
      node.querySelector('.person-role').textContent = person.label;
      return node.outerHTML;
    })
    .join('');

  return createLevelElement(level, peopleMarkup, sideClass, level.people.length);
}

function createSymbolLevel(level, sideClass) {
  const symbolMarkup = level.symbols
    .map((symbol) => `
      <div class="symbol-card">
        <div class="symbol-icon">${buildSymbol(symbol.type)}</div>
        <span class="symbol-role">${symbol.label}</span>
      </div>
    `)
    .join('');

  return createLevelElement(level, symbolMarkup, sideClass, level.symbols.length);
}

function createLevelElement(level, contentMarkup, sideClass, itemCount) {
  const section = document.createElement('article');
  section.className = `level ${sideClass}`;
  section.style.setProperty('--level-width', level.width);
  section.style.background = `linear-gradient(135deg, ${level.color}, ${shadeColor(level.color, -18)})`;
  applyLevelSizing(section, level.width, itemCount);

  section.innerHTML = `
    <div class="level-content">
      <div class="level-meta">
        <p class="level-index">${level.index}</p>
        <h2 class="level-title">${level.title}</h2>
        <p class="level-description">${level.description}</p>
      </div>
      <div class="people-row">${contentMarkup}</div>
    </div>
  `;

  return section;
}

function computeRowHeight(leftLevel, rightLevel) {
  const leftDensity = Math.max(leftLevel.people.length, 1);
  const rightDensity = Math.max(rightLevel.symbols.length, 1);
  const maxWidth = Math.max(leftLevel.width, rightLevel.width);
  const densityPenalty = Math.max(leftDensity, rightDensity) - 1;

  return Math.round(Math.max(112, 92 + maxWidth * 0.48 + densityPenalty * 12));
}

function initializeRuleToggle() {
  if (!ruleToggle) {
    return;
  }

  ruleToggle.addEventListener('click', () => {
    ruleModeIndex = (ruleModeIndex + 1) % ruleModes.length;
    ruleToggle.textContent = ruleModes[ruleModeIndex];
    ruleToggle.setAttribute('aria-label', `当前规则：${ruleModes[ruleModeIndex]}，点击切换`);
  });

  ruleToggle.setAttribute('aria-label', `当前规则：${ruleModes[ruleModeIndex]}，点击切换`);
}

function initializeClockTransition() {
  if (!clockTrigger) {
    return;
  }

  clockTrigger.addEventListener('click', () => {
    if (clockTrigger.classList.contains('is-shattering')) {
      return;
    }

    clockTrigger.classList.add('is-shattering');

    window.setTimeout(() => {
      window.location.href = './disorder.html';
    }, 3000);
  });
}

function applyLevelSizing(section, width, itemCount) {
  const widthScale = width / 90;
  const countPenalty = Math.max(0, itemCount - 1) * 0.07;
  const contentScale = Math.max(0.44, Math.min(1, widthScale + 0.08 - countPenalty));

  section.style.setProperty('--content-scale', contentScale.toFixed(2));

  if (width <= 50 || itemCount >= 3) {
    section.classList.add('compact-level');
  }
}

function buildAvatar(person) {
  const skin = '#ffd8b5';
  const hair = '#2f241f';
  const stroke = 'rgba(34, 34, 34, 0.18)';
  const outfit = person.color;
  const outfitShape = getOutfitShape(person.outfit, outfit, stroke);
  const accessory = getAccessory(person.accessory, outfit);

  return `
    <svg viewBox="0 0 88 104" width="100%" height="100%" aria-hidden="true">
      <ellipse cx="44" cy="100" rx="22" ry="4" fill="rgba(19, 28, 45, 0.14)" />
      ${accessory.back || ''}
      <circle cx="44" cy="18" r="11" fill="${skin}" />
      <path d="M33 18c1-9 20-14 24-2 1 3 0 7 0 7H33s-1-3 0-5z" fill="${hair}" />
      <rect x="39" y="28" width="10" height="8" rx="4" fill="${skin}" />
      ${outfitShape}
      <rect x="28" y="38" width="10" height="28" rx="5" fill="${skin}" stroke="${stroke}" />
      <rect x="50" y="38" width="10" height="28" rx="5" fill="${skin}" stroke="${stroke}" />
      <rect x="36" y="78" width="7" height="18" rx="3.5" fill="#2f4858" />
      <rect x="45" y="78" width="7" height="18" rx="3.5" fill="#2f4858" />
      <rect x="33" y="94" width="12" height="4" rx="2" fill="#18212f" />
      <rect x="44" y="94" width="12" height="4" rx="2" fill="#18212f" />
      ${accessory.front || ''}
    </svg>
  `;
}

function getOutfitShape(outfit, color, stroke) {
  const tie = `<path d="M43 38h3l3 8-4 10-4-10z" fill="#1f2937" />`;

  switch (outfit) {
    case 'suit':
    case 'formal':
    case 'blazer':
      return `
        <path d="M30 36h28l7 16-8 26H31L23 52z" fill="${color}" stroke="${stroke}" />
        <path d="M38 36l6 9 6-9" fill="#ffffff" opacity="0.92" />
        ${tie}
      `;
    case 'lab':
    case 'coat':
      return `
        <path d="M30 36h28l6 15-6 28H30L24 51z" fill="#f8fafc" stroke="${stroke}" />
        <path d="M39 36l5 8 5-8" fill="#dbeafe" />
        <rect x="33" y="50" width="8" height="11" rx="2" fill="${color}" />
        <rect x="47" y="50" width="8" height="11" rx="2" fill="${color}" />
      `;
    case 'vest':
      return `
        <path d="M30 36h28l7 16-8 26H31L23 52z" fill="#0f172a" stroke="${stroke}" />
        <path d="M37 36h14l5 40H32z" fill="${color}" />
      `;
    case 'shirt':
    case 'casual':
    case 'hoodie':
      return `
        <path d="M30 36h28l8 15-10 27H32L22 51z" fill="${color}" stroke="${stroke}" />
        <path d="M37 37c2 3 5 4 7 4s5-1 7-4" fill="none" stroke="#ffffff" stroke-width="2" opacity="0.8" />
      `;
    case 'apron':
      return `
        <path d="M30 36h28l8 15-9 27H31L22 51z" fill="#f3f4f6" stroke="${stroke}" />
        <path d="M36 42h16v34H36z" fill="${color}" rx="5" />
      `;
    case 'overall':
    case 'jumpsuit':
    case 'uniform':
      return `
        <path d="M30 36h28l7 14-8 28H31L23 50z" fill="${color}" stroke="${stroke}" />
        <rect x="38" y="44" width="12" height="10" rx="2" fill="#bfdbfe" opacity="0.9" />
      `;
    case 'jacket':
      return `
        <path d="M30 36h28l7 15-9 27H32L23 51z" fill="${color}" stroke="${stroke}" />
        <path d="M38 36l6 7 6-7" fill="#fde68a" opacity="0.85" />
      `;
    case 'field':
      return `
        <path d="M30 36h28l7 15-9 27H32L23 51z" fill="${color}" stroke="${stroke}" />
        <path d="M35 44h18v8H35z" fill="#854d0e" opacity="0.35" />
      `;
    case 'reflective':
      return `
        <path d="M30 36h28l7 15-9 27H32L23 51z" fill="${color}" stroke="${stroke}" />
        <path d="M33 50h22" stroke="#fef08a" stroke-width="4" />
        <path d="M35 58h18" stroke="#fef08a" stroke-width="4" />
      `;
    default:
      return `
        <path d="M30 36h28l7 15-9 27H32L23 51z" fill="${color}" stroke="${stroke}" />
      `;
  }
}

function getAccessory(accessory, color) {
  switch (accessory) {
    case 'badge':
      return {
        front: '<circle cx="55" cy="52" r="4" fill="#fef08a" /><path d="M55 56l-2 5 2-1 2 1-2-5z" fill="#facc15" />'
      };
    case 'tie':
      return {
        front: '<path d="M43 38h2l2 8-3 9-3-9z" fill="#0f172a" />'
      };
    case 'clipboard':
      return {
        front: '<rect x="55" y="50" width="11" height="15" rx="2" fill="#fde68a" stroke="#a16207" /><rect x="58" y="47" width="5" height="4" rx="1" fill="#a16207" />'
      };
    case 'tablet':
      return {
        front: '<rect x="56" y="48" width="12" height="16" rx="2" fill="#dbeafe" stroke="#1d4ed8" />'
      };
    case 'book':
      return {
        front: '<path d="M56 48h11a2 2 0 0 1 2 2v14H58a2 2 0 0 0-2 2z" fill="#fff7ed" stroke="#9a3412" />'
      };
    case 'nameTag':
      return {
        front: '<rect x="49" y="47" width="9" height="6" rx="2" fill="#ffffff" opacity="0.9" />'
      };
    case 'toolbelt':
      return {
        front: '<rect x="34" y="59" width="20" height="4" rx="2" fill="#422006" />'
      };
    case 'tray':
      return {
        front: '<ellipse cx="61" cy="50" rx="8" ry="3" fill="#e5e7eb" stroke="#6b7280" />'
      };
    case 'headset':
      return {
        front: '<path d="M35 18a9 9 0 0 1 18 0" fill="none" stroke="#0f172a" stroke-width="2" /><circle cx="33" cy="21" r="2.5" fill="#0f172a" /><circle cx="55" cy="21" r="2.5" fill="#0f172a" />'
      };
    case 'bag':
      return {
        back: '<rect x="21" y="43" width="12" height="18" rx="4" fill="#7c3aed" opacity="0.9" />'
      };
    case 'helmet':
      return {
        front: '<path d="M33 17a11 8 0 0 1 22 0v2H33z" fill="#facc15" />'
      };
    case 'cap':
      return {
        front: '<path d="M33 17c3-5 18-5 22 0v3H33z" fill="#1f2937" /><path d="M50 20h9" stroke="#1f2937" stroke-width="3" stroke-linecap="round" />'
      };
    case 'wrench':
      return {
        front: '<path d="M60 47l4 4-6 7-4-4z" fill="#cbd5e1" /><circle cx="65" cy="46" r="3" fill="none" stroke="#94a3b8" stroke-width="2" />'
      };
    case 'strawHat':
      return {
        front: '<ellipse cx="44" cy="17" rx="14" ry="4" fill="#eab308" /><path d="M36 12h16v7H36z" fill="#facc15" />'
      };
    case 'broom':
      return {
        front: '<rect x="62" y="45" width="2.5" height="22" rx="1" fill="#92400e" /><path d="M58 66h10l-2 8h-6z" fill="#fbbf24" />'
      };
    case 'gloves':
      return {
        front: '<circle cx="30" cy="63" r="3" fill="#bfdbfe" /><circle cx="58" cy="63" r="3" fill="#bfdbfe" />'
      };
    case 'backpack':
      return {
        back: '<rect x="22" y="42" width="12" height="18" rx="4" fill="#1d4ed8" opacity="0.85" />'
      };
    case 'scarf':
      return {
        front: '<path d="M39 34h10l-1 8h-8z" fill="#fef08a" /><rect x="46" y="35" width="3" height="16" rx="1.5" fill="#fef08a" />'
      };
    default:
      return { front: '', back: '' };
  }
}

function buildSymbol(type) {
  if (type === 'power') {
    return `
      <svg viewBox="0 0 88 104" width="100%" height="100%" aria-hidden="true">
        <ellipse cx="44" cy="98" rx="24" ry="5" fill="rgba(19, 28, 45, 0.14)" />
        <path d="M18 79h52v10H18z" fill="#4338ca" />
        <path d="M25 36h38l-5 19H30z" fill="#7c3aed" />
        <rect x="38" y="49" width="12" height="24" rx="4" fill="#6d28d9" />
        <path d="M24 33l7-14 13 7 13-7 7 14-8 8H32z" fill="#facc15" stroke="#a16207" stroke-width="2" />
        <circle cx="44" cy="26" r="6" fill="#fde68a" stroke="#a16207" stroke-width="2" />
        <path d="M44 8l2.8 6 6.7 0.8-5 4.4 1.4 6.5-5.9-3.3-5.9 3.3 1.4-6.5-5-4.4 6.7-0.8z" fill="#fde047" />
      </svg>
    `;
  }

  return `
    <svg viewBox="0 0 88 104" width="100%" height="100%" aria-hidden="true">
      <ellipse cx="44" cy="98" rx="24" ry="5" fill="rgba(19, 28, 45, 0.14)" />
      <circle cx="44" cy="52" r="23" fill="#f8fafc" stroke="#e2e8f0" stroke-width="3" />
      <circle cx="44" cy="52" r="19" fill="#facc15" stroke="#ca8a04" stroke-width="3" />
      <path d="M44 39v27" stroke="#7c2d12" stroke-width="4" stroke-linecap="round" />
      <path d="M51 42c-2-2-5-3-8-3-5 0-9 2.6-9 6.5 0 8 17 4.8 17 12.5 0 3.5-3.7 6-8 6-3.8 0-7.4-1.4-10-4" fill="none" stroke="#7c2d12" stroke-width="4" stroke-linecap="round" />
      <rect x="27" y="17" width="34" height="8" rx="4" fill="#16a34a" />
      <rect x="31" y="11" width="26" height="8" rx="4" fill="#22c55e" />
      <rect x="29" y="79" width="30" height="10" rx="5" fill="#16a34a" />
    </svg>
  `;
}

function shadeColor(hex, percent) {
  const numeric = hex.replace('#', '');
  const value = parseInt(numeric, 16);
  const adjust = Math.round(2.55 * percent);
  const r = Math.max(0, Math.min(255, (value >> 16) + adjust));
  const g = Math.max(0, Math.min(255, ((value >> 8) & 0x00ff) + adjust));
  const b = Math.max(0, Math.min(255, (value & 0x0000ff) + adjust));
  return `#${(0x1000000 + r * 0x10000 + g * 0x100 + b).toString(16).slice(1)}`;
}

renderLegend();
renderSymmetricPyramids();
initializeRuleToggle();
initializeClockTransition();
