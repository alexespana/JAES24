import { Injectable } from '@angular/core';
import { API_ROUTES } from '../../api-routes';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';


interface Repository {
  repository: string;
  branch: string;
  features_file: string;
  pickle_pattern: string;
  available: string;
}

@Injectable({
  providedIn: 'root'
})
export class RepositoriesService {

  private availableModelsUrl = API_ROUTES.AVAILABLEMODELS;

  constructor(private http: HttpClient) { }

  getAvailableModels(): Observable<Repository[]> {
    return this.http.get<Repository[]>(this.availableModelsUrl);
  }
}
