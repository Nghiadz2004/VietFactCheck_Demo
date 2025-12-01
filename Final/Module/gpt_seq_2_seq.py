import json, re

def gpt_call(system_prompt, input_prompt,
             client,
             model="gpt-5-nano-2025-08-07",
             flag='list'):

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_prompt},
        ]
    )

    # SDK mới trả về object, không phải dict
    content = response.choices[0].message.content.strip()

    if flag == 'raw_text':
      # Cố gắng parse JSON
      try:
          expanded = json.loads(content)
          if not isinstance(expanded, list):
              raise ValueError
      except json.JSONDecodeError:
          m = re.search(r'\[.*\]', content, re.S)
          if m:
              try:
                  expanded = json.loads(m.group(0))
                  if isinstance(expanded, list):
                      return expanded
              except json.JSONDecodeError:
                  # Nếu chuỗi regex tìm được cũng không phải JSON, chuyển sang trả về text thô
                  pass

      return [content]
    else:
      # Cố gắng parse JSON
      try:
          expanded = json.loads(content)
          if not isinstance(expanded, list):
              raise ValueError
      except Exception:
          m = re.search(r'\[.*\]', content, re.S)
          expanded = json.loads(m.group(0)) if m else [input_prompt]

      return expanded