// A self-reference passes the host value through without creating an empty
// variable when it is absent. Other references retain their usual expansion.
export function isEnvPassthrough(key, value) {
  return value === `$${key}` || value === `\${${key}}`;
}

export function envToExports(env) {
  return Object.entries(env || {}).map(([key, value]) =>
    isEnvPassthrough(key, value)
      ? `if [ "\${${key}+x}" = x ]; then export ${key}; fi`
      : `export ${key}=${value}`
  ).join("\n");
}
