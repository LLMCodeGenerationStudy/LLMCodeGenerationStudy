def is_byte_token(s: str) -> bool:
  """Returns True if s is a byte string like "<0xAB>".
    input:
        s: string
    return
        bool: Ture or False
  """
  # Bytes look like "<0xAB>".
  # ----
  
  if len(s) != 6 or s[0:3] != "<0x" or s[-1] != ">":
    return False
  return True


# unit test cases
print(is_byte_token("<0xAB>"))
print(is_byte_token("<0xG1>"))
print(is_byte_token("0xAB>"))
print(is_byte_token("<0xAB"))
