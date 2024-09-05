import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AvailableModelsComponent } from './available-models.component';

describe('AvailableModelsComponent', () => {
  let component: AvailableModelsComponent;
  let fixture: ComponentFixture<AvailableModelsComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AvailableModelsComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(AvailableModelsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
