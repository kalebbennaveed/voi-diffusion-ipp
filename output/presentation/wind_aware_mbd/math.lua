-- Preserve editable LaTeX and render it with the embedded local KaTeX library.
function Math(el)
  local text = el.text:gsub('&','&amp;'):gsub('<','&lt;'):gsub('>','&gt;'):gsub('"','&quot;')
  local mode = el.mathtype == 'DisplayMath' and 'display' or 'inline'
  return pandoc.RawInline('html', '<span class="latex-' .. mode .. '" data-latex="' .. text .. '"></span>')
end
