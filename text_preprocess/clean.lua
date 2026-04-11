-- 1. 移除图片及图片容器（兼容新版 Pandoc 的 Figure 节点）
function Image(el) return {} end
function Figure(el) return {} end

-- 2. 移除脚注/内联注释标记（如 [^1]、[1] 等）
function Note(el) return {} end

-- 3. 仅保留 1~3 级标题，超出层级的自动转为普通段落
function Header(el)
  if el.level > 3 then
    return pandoc.Para(el.content)
  end
  return el
end

-- 4. 清理因移除元素产生的空段落
function Para(el)
  if #el.content == 0 then return {} end
  return el
end