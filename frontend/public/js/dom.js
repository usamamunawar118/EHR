// Tiny element builder. Text is always inserted as text nodes (never as HTML), so content that
// originates in records or in an LLM can never inject markup or script.

/**
 * h('p', { class: 'x', onclick: fn }, 'text', otherNode, [more, children])
 */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs ?? {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') el.className = value;
    else if (key.startsWith('on') && typeof value === 'function') {
      el.addEventListener(key.slice(2).toLowerCase(), value);
    } else el.setAttribute(key, value === true ? '' : String(value));
  }
  append(el, children);
  return el;
}

export function append(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function clear(el) {
  el.replaceChildren();
  return el;
}

/** Announce a message to screen readers via the page's polite live region. */
export function announce(text) {
  const region = document.getElementById('status');
  if (region) region.textContent = text;
}
