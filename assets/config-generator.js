/**
 * vLLM Deployment Config Generator
 *
 * A vanilla JS interactive configuration selector for vLLM deployment recipes.
 * Recipe authors define a config object and call ConfigGenerator.create(element, config).
 *
 * Config schema:
 * {
 *   options: {
 *     optionKey: {
 *       name: 'optionKey',
 *       title: 'Display Title',
 *       items: [{ id: 'val', label: 'Label', subtitle: '', default: true, disabled: false, disabledReason: '' }],
 *       getDynamicItems: (values) => items[],    // optional: compute items from current state
 *       condition: (values) => boolean,           // optional: show/hide this option
 *       commandRule: (value) => string | null,    // optional: simple flag contribution
 *     },
 *     ...
 *   },
 *   generateCommand: function(values) { return 'shell command'; }
 * }
 */
class ConfigGenerator {
  constructor(container, config) {
    this.container = container;
    this.config = config;
    this.values = this._getInitialState();
    this._render();
  }

  static create(element, config) {
    return new ConfigGenerator(element, config);
  }

  _getInitialState() {
    const state = {};
    for (const [key, option] of Object.entries(this.config.options)) {
      let items = option.items;
      if (option.getDynamicItems) {
        // Bootstrap: build default values from static items first
        const defaults = {};
        for (const [k, opt] of Object.entries(this.config.options)) {
          if (opt.items && opt.items.length > 0) {
            const d = opt.items.find(i => i.default);
            defaults[k] = d ? d.id : opt.items[0].id;
          }
        }
        items = option.getDynamicItems(defaults);
      }
      const defaultItem = items && items.find(i => i.default);
      state[key] = defaultItem ? defaultItem.id : (items && items[0] ? items[0].id : '');
    }
    return state;
  }

  _handleChange(optionName, value) {
    this.values[optionName] = value;
    this._render();
  }

  _render() {
    this.container.innerHTML = '';
    const wrapper = document.createElement('div');
    wrapper.className = 'config-generator';

    for (const [key, option] of Object.entries(this.config.options)) {
      // Conditional visibility
      if (option.condition && !option.condition(this.values)) {
        continue;
      }

      const card = document.createElement('div');
      card.className = 'config-option';

      const title = document.createElement('div');
      title.className = 'config-option-title';
      title.textContent = option.title;
      card.appendChild(title);

      const items = document.createElement('div');
      items.className = 'config-option-items';

      const resolvedItems = option.getDynamicItems
        ? option.getDynamicItems(this.values)
        : option.items;

      // If the current value is disabled or doesn't exist in resolved items, auto-select first enabled
      const currentValid = resolvedItems.find(i => i.id === this.values[key] && !i.disabled);
      if (!currentValid) {
        const firstEnabled = resolvedItems.find(i => !i.disabled);
        if (firstEnabled) {
          this.values[key] = firstEnabled.id;
        }
      }

      for (const item of resolvedItems) {
        const label = document.createElement('label');
        const isChecked = this.values[key] === item.id;
        const isDisabled = item.disabled || false;

        label.className = 'config-option-label'
          + (isChecked ? ' checked' : '')
          + (isDisabled ? ' disabled' : '');
        if (item.disabledReason) {
          label.title = item.disabledReason;
        }

        const input = document.createElement('input');
        input.type = 'radio';
        input.name = option.name;
        input.value = item.id;
        input.checked = isChecked;
        input.disabled = isDisabled;
        if (!isDisabled) {
          input.addEventListener('change', () => this._handleChange(key, item.id));
        }
        label.appendChild(input);

        label.appendChild(document.createTextNode(item.label));

        if (item.subtitle) {
          const sub = document.createElement('small');
          sub.className = 'config-option-subtitle';
          sub.textContent = item.subtitle;
          label.appendChild(sub);
        }

        items.appendChild(label);
      }

      card.appendChild(items);
      wrapper.appendChild(card);
    }

    // Command output
    const cmdCard = document.createElement('div');
    cmdCard.className = 'config-command';

    const cmdTitle = document.createElement('div');
    cmdTitle.className = 'config-command-title';
    cmdTitle.textContent = 'Run this Command:';
    cmdCard.appendChild(cmdTitle);

    const cmdPre = document.createElement('pre');
    cmdPre.className = 'config-command-display';
    const command = this.config.generateCommand
      ? this.config.generateCommand(this.values)
      : '';
    cmdPre.textContent = command;

    // Copy button
    const copyBtn = document.createElement('button');
    copyBtn.className = 'config-command-copy';
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(command).then(() => {
        copyBtn.textContent = 'Copied!';
        setTimeout(() => { copyBtn.textContent = 'Copy'; }, 1500);
      });
    });
    cmdPre.appendChild(copyBtn);

    cmdCard.appendChild(cmdPre);
    wrapper.appendChild(cmdCard);

    this.container.appendChild(wrapper);
  }
}

// Expose globally for use in markdown <script> blocks
window.ConfigGenerator = ConfigGenerator;
